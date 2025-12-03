import contextlib
import torch
import re
from utils import tokenize_prefix_and_target
from peft import LoraConfig , get_peft_model
from baselines.mend.algs.mend_rawdoc import MEND_RAWDOC
from baselines.mend.algs.mend_augdoc import MEND_AUGDOC
from baselines.mend.algs.mend import MEND
from baselines.memit.memit_main import apply_memit_to_model
from baselines.alphaEdit.AlphaEdit_main import apply_AlphaEdit_to_model
from baselines.memit.rome.rome_main import apply_rome_to_model
from copy import deepcopy
import transformers
import hydra
import os
from omegaconf import OmegaConf
from collections import OrderedDict
from baselines.alphaEdit.util import nethook
from baselines.alphaEdit.AlphaEdit_main import get_cov


class EditWrapper :
    def __init__(self , model , tokenizer , config) :
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def edit(self , edits , is_sequential , logging) :
        raise NotImplementedError
    def autorestore(self) :
        raise NotImplementedError


class NoEdit(EditWrapper) :
    def __init__(self , model , tokenizer , config) :
        super().__init__(model , tokenizer , config)

    def edit(self , edits , is_sequential , logging) :
        return {}

    @contextlib.contextmanager
    def autorestore(self) :
        try :
            yield {}
        finally :
            pass


class FineTuning(EditWrapper) :

    def __init__(self , model , tokenizer , config) :
        super().__init__(model , tokenizer , config)
        if hasattr(self.config.editor , "trainable_pattern") and "{layers}" in self.config.editor.trainable_pattern :
            if self.config.editor.layers is not None :
                layers = []
                for l in str(config.editor.layers).split(",") :
                    if "-" in l :
                        start , _ , end = l.split("-")
                        layers += [str(i) for i in range(int(start) , int(end) + 1)]
                    else :
                        layers.append(l)
                layers = "(" + "|".join(layers) + ")"
            else :
                layers = "."
            self.config.editor.trainable_pattern = config.editor.trainable_pattern.replace("{layers}" , layers)

    def find_trainable_parameters(self) :
        self.model.requires_grad_(False)
        trainable = {}
        for n , p in self.model.named_parameters() :
            if re.search(self.config.editor.trainable_pattern , n) :
                p.requires_grad_(True)
                trainable[n] = p
        return trainable

    @contextlib.contextmanager
    def autorestore(self) :
        state_backup = self.find_trainable_parameters()
        if self.config.low_vram :
            state_backup = {k : v.detach().clone().to(device = 'cpu') for k , v in state_backup.items()}
        else :
            state_backup = {k : v.detach().clone() for k , v in state_backup.items()}
        try :
            yield state_backup
        finally :
            self.model.load_state_dict(state_backup , strict = False)

    def collate_fn(self , edits) :
        prefixes = []
        targets = []
        for e in edits :
            if "doc" in e :
                prefixes.append("")
                targets.append(e['doc'])
            elif 'input' in e :
                prefixes.append(e['input'])
                targets.append(e['target'])
        return tokenize_prefix_and_target(self.tokenizer , prefixes , targets)

    def cut_minibatch(self , batch , max_minibatch_size = None) :
        if max_minibatch_size is None and self.config.editor.minibatch_tokens > 0 :
            input_length = batch['input_ids'].size(1)
            max_minibatch_size = max(self.config.editor.minibatch_tokens // input_length , 1)
        if max_minibatch_size is not None :
            minibatches = [transformers.BatchEncoding(
                {k : v[i :i + max_minibatch_size] for k , v in batch.items()})
                for i in range(0 , batch['input_ids'].size(0) , max_minibatch_size)]
            return minibatches
        else :
            return [batch]

    def edit(self , edits , is_sequential , logging) :
        if self.config.editor.train_mode :
            self.model.train()
        else :
            self.model.eval()
        torch.cuda.empty_cache()
        trainable_parameters = self.find_trainable_parameters()
        optimizer = torch.optim.__dict__.get(self.config.editor.opt_name)(
            trainable_parameters.values() ,
            **dict(self.config.editor.opt_kwargs)
        )
        if getattr(self.config.editor , "ict_distill" , False) :
            edit_batch , aug_batch , ict_batch = self.ict_distill_collate_fn(edits)
            if edit_batch is not None :
                all_edit_target_tokens = (edit_batch['labels'][: , 1 :] != -100).sum().detach().item()
                edit_minibatches = self.cut_minibatch(edit_batch)
            else :
                edit_minibatches = []
            if aug_batch is not None :
                all_aug_target_tokens = (aug_batch['labels'][: , 1 :] != -100).sum().detach().item()
                ict_minibatches = self.cut_minibatch(ict_batch)
                aug_minibatches = self.cut_minibatch(aug_batch , max_minibatch_size = ict_minibatches[0].input_ids.size(0))
            else :
                aug_minibatches = []
                ict_minibatches = []
            ict_outputs_for_minibatches = []
            for aug_minibatch , ict_minibatch in zip(aug_minibatches , ict_minibatches) :
                aug_minibatch = aug_minibatch.to(self.model.device)
                ict_minibatch = ict_minibatch.to(self.model.device)
                with torch.no_grad() :
                    ict_outputs = self.model(**ict_minibatch).logits[: , :-1][ict_minibatch['labels'][: , 1 :] != -100] 
                    aug_outputs = self.model(**aug_minibatch).logits[: , :-1][aug_minibatch['labels'][: , 1 :] != -100]
                    ict_outputs = ict_outputs + self.config.editor.ict_contra_coeff * (ict_outputs - aug_outputs) 
                    ict_outputs = ict_outputs.detach()
                    ict_outputs_for_minibatches.append(ict_outputs)
            for _ in range(self.config.editor.steps) :
                optimizer.zero_grad()
                loss = 0.
                for minibatch in edit_minibatches :
                    target_tokens = (minibatch['labels'][: , 1 :] != -100).sum().detach().item()
                    miniloss = self.config.editor.edit_loss_coeff * self.model(**minibatch.to(device = self.model.device)).loss * (target_tokens / all_edit_target_tokens)
                    miniloss.backward()
                    loss += miniloss.detach().clone()
                for aug_minibatch , ict_minibatch , ict_outputs in zip(aug_minibatches , ict_minibatches , ict_outputs_for_minibatches) :
                    aug_minibatch = aug_minibatch.to(self.model.device)
                    target_tokens = (aug_minibatch["labels"][: , 1 :] != -100).sum().detach().item()
                    aug_results = self.model(**aug_minibatch)
                    target_loss = aug_results.loss
                    aug_outputs = aug_results.logits[: , :-1][aug_minibatch['labels'][: , 1 :] != -100]
                    kl_loss = (
                            ict_outputs.softmax(-1) * (ict_outputs.log_softmax(-1) - aug_outputs.log_softmax(-1))
                    ).sum(-1).mean()
                    miniloss = self.config.editor.aug_loss_coeff * (
                            (self.config.editor.kl_coeff) * kl_loss + (1 - self.config.editor.kl_coeff) * target_loss) * target_tokens / all_aug_target_tokens
                    miniloss.backward()
                    loss += miniloss.detach().clone()
                logging.info(f' epoch{_}:loss:{loss.item()}')
                if loss < self.config.editor.early_stop :
                    optimizer.zero_grad()
                    break
                optimizer.step()
        else :
            batch = self.collate_fn(edits)
            all_target_tokens = (batch['labels'] != -100).sum().detach().item()
            minibatches = self.cut_minibatch(batch)
            for _ in range(self.config.editor.steps) :
                optimizer.zero_grad()
                loss = 0.
                for minibatch in minibatches :
                    target_tokens = (minibatch['labels'] != -100).sum().detach().item()
                    miniloss = self.model(**minibatch.to(device = self.model.device)).loss * (target_tokens / all_target_tokens)
                    miniloss.backward()
                    loss += miniloss.detach().clone()
                logging.info(f'epoch{_}:loss:{loss.item()}')
                if loss < self.config.editor.early_stop :
                    optimizer.zero_grad()
                    break
                optimizer.step()
                torch.cuda.empty_cache()
        return edits


class EvoEdit(EditWrapper) :
    def __init__(self , model , tokenizer , config) :
        super().__init__(model , tokenizer , config)
        if hasattr(self.config.editor , "trainable_pattern") and "{layers}" in self.config.editor.trainable_pattern :
            if self.config.editor.layers is not None :
                layers = []
                for l in str(config.editor.layers).split(",") :
                    if "-" in l :
                        start , _ , end = l.split("-")
                        layers += [str(i) for i in range(int(start) , int(end) + 1)]
                    else :
                        layers.append(l)
                layers = "(" + "|".join(layers) + ")"
            else :
                layers = "."
            self.config.editor.trainable_pattern = config.editor.trainable_pattern.replace("{layers}" , layers)

    def find_trainable_parameters(self) :
        self.model.requires_grad_(False)
        trainable = {}
        for n , p in self.model.named_parameters() :
            if re.search(self.config.editor.trainable_pattern , n) :
                p.requires_grad_(True)
                trainable[n] = p
        return trainable

    @contextlib.contextmanager
    def autorestore(self) :
        state_backup = self.find_trainable_parameters()
        if self.config.low_vram :
            state_backup = {k : v.detach().clone().to(device = 'cpu') for k , v in state_backup.items()}
        else :
            state_backup = {k : v.detach().clone() for k , v in state_backup.items()}
        try :
            yield state_backup
        finally :
            self.model.load_state_dict(state_backup , strict = False)

    def collate_fn(self , edits) :
        prefixes = []
        targets = []
        for e in edits :
            if "doc" in e :
                prefixes.append("")
                targets.append(e['doc'])
            elif 'input' in e :
                prefixes.append(e['input'])
                targets.append(e['target'])
        return tokenize_prefix_and_target(self.tokenizer , prefixes , targets)

    def cut_minibatch(self , batch , max_minibatch_size = None) :
        if max_minibatch_size is None and self.config.editor.minibatch_tokens > 0 :
            input_length = batch['input_ids'].size(1)
            max_minibatch_size = max(self.config.editor.minibatch_tokens // input_length , 1)
        if max_minibatch_size is not None :
            minibatches = [transformers.BatchEncoding({k : v[i :i + max_minibatch_size] for k , v in batch.items()}) for i in range(0 , batch['input_ids'].size(0) , max_minibatch_size)]
            return minibatches
        else :
            return [batch]

    def edit(self , edits , is_sequential , logging) :
        if self.config.editor.train_mode :
            self.model.train()
        else :
            self.model.eval()
        trainable_parameters = self.find_trainable_parameters()
        optimizer = torch.optim.__dict__.get(self.config.editor.opt_name)(
            trainable_parameters.values() ,
            **dict(self.config.editor.opt_kwargs)
        )
        batch = self.collate_fn(edits)
        all_target_tokens = (batch['labels'] != -100).sum().detach().item()
        minibatches = self.cut_minibatch(batch)
        hook_handle = None
        if self.model.training and hook_handle is None :
            hook_handle = self.model.base_model.get_input_embeddings().register_forward_hook(self.forward_hook)
            print("add noise hook")
        elif not self.model.training and hook_handle is not None :
            hook_handle.remove()
            hook_handle = None
            print("remove noise hook")
        importance_score_q_proj = torch.zeros(self.model.config.num_hidden_layers , 1).to('cpu')
        importance_score_k_proj = torch.zeros(self.model.config.num_hidden_layers , 1).to('cpu')
        importance_score_v_proj = torch.zeros(self.model.config.num_hidden_layers , 1).to('cpu')
        importance_score_o_proj = torch.zeros(self.model.config.num_hidden_layers , 1).to('cpu')
        gate_importance_score = torch.zeros(self.model.config.num_hidden_layers , 1).to('cpu')
        up_importance_score = torch.zeros(self.model.config.num_hidden_layers , 1).to('cpu')
        down_importance_score = torch.zeros(self.model.config.num_hidden_layers , 1).to('cpu')
        for _ in range(self.config.editor.steps) :
            optimizer.zero_grad()
            loss = 0.
            for minibatch in minibatches :
                target_tokens = (minibatch['labels'] != -100).sum().detach().item()
                miniloss = self.model(**minibatch.to(device = self.model.device)).loss * (target_tokens / all_target_tokens)
                miniloss.backward()
                loss += miniloss.detach().clone()
            logging.info(f' epoch{_}:loss:{loss.item()}')
            if self.model.name_or_path == 'gpt2-xl' :
                for layer in range(self.model.config.num_hidden_layers) :
                    self_attention = self.model.transformer.h[layer].attn
                    hidden_size = self_attention.c_attn.weight.shape[0] // 3
                    attn_q = self_attention.c_attn.weight[:hidden_size , :].clone().detach()
                    grad_attn_q = self_attention.c_attn.weight.grad[:hidden_size , :]
                    attn_k = self_attention.c_attn.weight[hidden_size :2 * hidden_size , :].clone().detach()
                    grad_attn_k = self_attention.c_attn.weight.grad[hidden_size :2 * hidden_size , :]
                    attn_v = self_attention.c_attn.weight[2 * hidden_size : , :].clone().detach()
                    grad_attn_v = self_attention.c_attn.weight.grad[2 * hidden_size : , :]
                    attn_o = self_attention.c_proj.weight.clone().detach()
                    grad_attn_o = self_attention.c_proj.weight.grad
                    dot_q_proj = torch.einsum("hl,hl->h" , [grad_attn_q.to("cpu") , attn_q.to("cpu")]).to('cpu')   
                    dot_k_proj = torch.einsum("hl,hl->h" , [grad_attn_k.to("cpu") , attn_k.to("cpu")]).to('cpu')   
                    dot_v_proj = torch.einsum("hl,hl->h" , [grad_attn_v.to("cpu") , attn_v.to("cpu")]).to('cpu')   
                    dot_o_proj = torch.einsum("hl,hl->h" , [grad_attn_o.to("cpu") , attn_o.to("cpu")]).to('cpu')   
                    importance_score_q_proj[layer] += dot_q_proj.abs().sum().detach()
                    importance_score_k_proj[layer] += dot_k_proj.abs().sum().detach()
                    importance_score_v_proj[layer] += dot_v_proj.abs().sum().detach()
                    importance_score_o_proj[layer] += dot_o_proj.abs().sum().detach()
                    self_mlp = self.model.transformer.h[layer].mlp
                    up = self_mlp.c_fc.weight.clone().detach()
                    grad_up = self_mlp.c_fc.weight.grad
                    down = self_mlp.c_proj.weight.clone().detach()
                    grad_down = self_mlp.c_proj.weight.grad
                    dot_up = torch.einsum("hl,hl->h" , [grad_up.to("cpu") , up.to("cpu")]).to('cpu')   
                    dot_down = torch.einsum("hl,hl->h" , [grad_down.to("cpu") , down.to("cpu")]).to('cpu')   
                    up_importance_score[layer] += dot_up.abs().sum().detach()
                    down_importance_score[layer] += dot_down.abs().sum().detach()
            elif self.model.name_or_path == 'EleutherAI/gpt-j-6B' :
                for layer in range(self.model.config.n_layer) :
                    self_attention = self.model.transformer.h[layer].attn
                    attn_q = self_attention.q_proj.weight.clone().detach()    
                    grad_attn_q = self_attention.q_proj.weight.grad    
                    attn_k = self_attention.k_proj.weight.clone().detach()    
                    grad_attn_k = self_attention.k_proj.weight.grad    
                    attn_v = self_attention.v_proj.weight.clone().detach()    
                    grad_attn_v = self_attention.v_proj.weight.grad    
                    attn_o = self_attention.out_proj.weight.clone().detach()    
                    grad_attn_o = self_attention.out_proj.weight.grad    
                    dot_q_proj = torch.einsum("hl,hl->h" , [grad_attn_q.to("cpu") , attn_q.to("cpu")]).to('cpu')  
                    dot_k_proj = torch.einsum("hl,hl->h" , [grad_attn_k.to("cpu") , attn_k.to("cpu")]).to('cpu')   
                    dot_v_proj = torch.einsum("hl,hl->h" , [grad_attn_v.to("cpu") , attn_v.to("cpu")]).to('cpu')   
                    dot_o_proj = torch.einsum("hl,hl->h" , [grad_attn_o.to("cpu") , attn_o.to("cpu")]).to('cpu')   
                    importance_score_q_proj[layer] += dot_q_proj.abs().sum().detach()
                    importance_score_k_proj[layer] += dot_k_proj.abs().sum().detach()
                    importance_score_v_proj[layer] += dot_v_proj.abs().sum().detach()
                    importance_score_o_proj[layer] += dot_o_proj.abs().sum().detach()
                    self_mlp = self.model.transformer.h[layer].mlp
                    up = self_mlp.fc_in.weight.clone().detach()
                    grad_up = self_mlp.fc_in.weight.grad
                    down = self_mlp.fc_out.weight.clone().detach()
                    grad_down = self_mlp.fc_out.weight.grad
                    dot_up = torch.einsum("hl,hl->h" , [grad_up.to("cpu") , up.to("cpu")]).to('cpu')   
                    dot_down = torch.einsum("hl,hl->h" , [grad_down.to("cpu") , down.to("cpu")]).to('cpu')   
                    up_importance_score[layer] += dot_up.abs().sum().detach()
                    down_importance_score[layer] += dot_down.abs().sum().detach()
            elif self.model.name_or_path == 'meta-llama/Llama-3.1-8B':
                for layer in range(self.model.config.num_hidden_layers) :
                    self_attention = self.model.model.layers[layer].self_attn
                    attn_q = self_attention.q_proj.weight.clone().detach()    
                    grad_attn_q = self_attention.q_proj.weight.grad    
                    attn_k = self_attention.k_proj.weight.clone().detach()    
                    grad_attn_k = self_attention.k_proj.weight.grad    
                    attn_v = self_attention.v_proj.weight.clone().detach()    
                    grad_attn_v = self_attention.v_proj.weight.grad    
                    attn_o = self_attention.o_proj.weight.clone().detach()    
                    grad_attn_o = self_attention.o_proj.weight.grad
                    dot_q_proj = torch.einsum("hl,hl->h" , [grad_attn_q.to("cpu") , attn_q.to("cpu")]).to('cpu')   
                    dot_k_proj = torch.einsum("hl,hl->h" , [grad_attn_k.to("cpu") , attn_k.to("cpu")]).to('cpu')   
                    dot_v_proj = torch.einsum("hl,hl->h" , [grad_attn_v.to("cpu") , attn_v.to("cpu")]).to('cpu')   
                    dot_o_proj = torch.einsum("hl,hl->h" , [grad_attn_o.to("cpu") , attn_o.to("cpu")]).to('cpu')
                    importance_score_q_proj[layer] += dot_q_proj.abs().sum().detach()
                    importance_score_k_proj[layer] += dot_k_proj.abs().sum().detach()
                    importance_score_v_proj[layer] += dot_v_proj.abs().sum().detach()
                    importance_score_o_proj[layer] += dot_o_proj.abs().sum().detach()
                    self_mlp = self.model.model.layers[layer].mlp
                    gate = self_mlp.gate_proj.weight.clone().detach()
                    grad_gate = self_mlp.gate_proj.weight.grad
                    up = self_mlp.up_proj.weight.clone().detach()
                    grad_up = self_mlp.up_proj.weight.grad
                    down = self_mlp.down_proj.weight.clone().detach()
                    grad_down = self_mlp.down_proj.weight.grad
                    dot_up = torch.einsum("hl,hl->h" , [grad_up.to("cpu") , up.to("cpu")]).to('cpu')   
                    dot_gate = torch.einsum("hl,hl->h" , [grad_gate.to("cpu") , gate.to("cpu")]).to('cpu')   
                    dot_down = torch.einsum("hl,hl->h" , [grad_down.to("cpu") , down.to("cpu")]).to('cpu')   
                    gate_importance_score[layer] += dot_gate.abs().sum().detach()
                    up_importance_score[layer] += dot_up.abs().sum().detach()
                    down_importance_score[layer] += dot_down.abs().sum().detach()
            else :
                for layer in range(self.model.config.num_hidden_layers) :
                    self_attention = self.model.get_decoder().layers[layer].self_attn
                    attn_q = self_attention.q_proj.weight.clone().detach()    
                    grad_attn_q = self_attention.q_proj.weight.grad    
                    attn_k = self_attention.k_proj.weight.clone().detach()    
                    grad_attn_k = self_attention.k_proj.weight.grad    
                    attn_v = self_attention.v_proj.weight.clone().detach()    
                    grad_attn_v = self_attention.v_proj.weight.grad    
                    attn_o = self_attention.o_proj.weight.clone().detach()    
                    grad_attn_o = self_attention.o_proj.weight.grad
                    dot_q_proj = torch.einsum("hl,hl->h" , [grad_attn_q.to("cpu") , attn_q.to("cpu")]).to('cpu')   
                    dot_k_proj = torch.einsum("hl,hl->h" , [grad_attn_k.to("cpu") , attn_k.to("cpu")]).to('cpu')   
                    dot_v_proj = torch.einsum("hl,hl->h" , [grad_attn_v.to("cpu") , attn_v.to("cpu")]).to('cpu')   
                    dot_o_proj = torch.einsum("hl,hl->h" , [grad_attn_o.to("cpu") , attn_o.to("cpu")]).to('cpu')
                    importance_score_q_proj[layer] += dot_q_proj.abs().sum().detach()
                    importance_score_k_proj[layer] += dot_k_proj.abs().sum().detach()
                    importance_score_v_proj[layer] += dot_v_proj.abs().sum().detach()
                    importance_score_o_proj[layer] += dot_o_proj.abs().sum().detach()
                    self_mlp = self.model.get_decoder().layers[layer].mlp
                    gate = self_mlp.gate_proj.weight.clone().detach()
                    grad_gate = self_mlp.gate_proj.weight.grad
                    up = self_mlp.up_proj.weight.clone().detach()
                    grad_up = self_mlp.up_proj.weight.grad
                    down = self_mlp.down_proj.weight.clone().detach()
                    grad_down = self_mlp.down_proj.weight.grad
                    dot_up = torch.einsum("hl,hl->h" , [grad_up.to("cpu") , up.to("cpu")]).to('cpu')   
                    dot_gate = torch.einsum("hl,hl->h" , [grad_gate.to("cpu") , gate.to("cpu")]).to('cpu')   
                    dot_down = torch.einsum("hl,hl->h" , [grad_down.to("cpu") , down.to("cpu")]).to('cpu')   
                    gate_importance_score[layer] += dot_gate.abs().sum().detach()
                    up_importance_score[layer] += dot_up.abs().sum().detach()
                    down_importance_score[layer] += dot_down.abs().sum().detach()
            if loss < self.config.editor.early_stop :
                optimizer.zero_grad()
                hook_handle.remove()
                hook_handle = None
                break
            optimizer.step()
        if self.model.name_or_path == 'gpt2-xl' or self.model.name_or_path == 'EleutherAI/gpt-j-6B' :
            del attn_q , grad_attn_q , attn_k , grad_attn_k , attn_v , grad_attn_v , attn_o , grad_attn_o , dot_q_proj , dot_k_proj , dot_v_proj , dot_o_proj
            del up , grad_up , down , grad_down , dot_up , dot_down
            importance = torch.stack([importance_score_q_proj , importance_score_k_proj , importance_score_v_proj , importance_score_o_proj , gate_importance_score , up_importance_score , down_importance_score] , dim = 0)
            componentdict = {0 : "q_proj" , 1 : "k_proj" , 2 : "v_proj" , 3 : "o_proj" , 4 : "up_proj" , 5 : "down_proj"}
            total_importance = {}
            for i in range(6) :
                for j in range(len(importance_score_k_proj)) :
                    if i >= 4 :
                        total_importance["layers." + str(j) + ".mlp." + componentdict[i]] = importance[i][j].item()
                    else :
                        total_importance["layers." + str(j) + ".self_attn." + componentdict[i]] = importance[i][j].item()
            sorted_importance = OrderedDict(sorted(total_importance.items() , key = lambda x : x[1] , reverse = True))
            importance_modules = list(sorted_importance.keys())[:int(len(sorted_importance) * self.config.editor.importance_ratio)]
        else :
            del attn_q , grad_attn_q , attn_k , grad_attn_k , attn_v , grad_attn_v , attn_o , grad_attn_o , dot_q_proj , dot_k_proj , dot_v_proj , dot_o_proj
            del up , grad_up , gate , grad_gate , down , grad_down , dot_up , dot_gate , dot_down
            importance = torch.stack([importance_score_q_proj , importance_score_k_proj , importance_score_v_proj , importance_score_o_proj , gate_importance_score , up_importance_score , down_importance_score] , dim = 0)
            componentdict = {0 : "q_proj" , 1 : "k_proj" , 2 : "v_proj" , 3 : "o_proj" , 4 : "gate_proj" , 5 : "up_proj" , 6 : "down_proj"}
            total_importance = {}
            for i in range(7) :
                for j in range(len(importance_score_k_proj)) :
                    if i >= 4 :
                        total_importance["layers." + str(j) + ".mlp." + componentdict[i]] = importance[i][j].item()
                    else :
                        total_importance["layers." + str(j) + ".self_attn." + componentdict[i]] = importance[i][j].item()
            sorted_importance = OrderedDict(sorted(total_importance.items() , key = lambda x : x[1] , reverse = True))
            importance_modules = list(sorted_importance.keys())[:int(len(sorted_importance) * self.config.editor.importance_ratio)]
        logging.info(f'importance_ratio:{self.config.editor.importance_ratio},importance_modules: {len(importance_modules)}')
        return edits,importance_modules

    def forward_hook(self , embedding_obj , input , output) :
        perturbation_alpha = self.model.perturbation_alpha
        dims = torch.tensor(output.size(1) * output.size(2))
        mag_norm = perturbation_alpha / torch.sqrt(dims)
        if perturbation_alpha ==0:
            output = output
        else:
            output = output + torch.zeros_like(output).uniform_(-mag_norm , mag_norm)
        return output

class MENDWrapper(EditWrapper) :
    def __init__(self , model , tokenizer , config) :
        super().__init__(model , tokenizer , config)
        self.original_model = model
        self.init_model(model , tokenizer , config.editor.hparams)

    def init_model(self , model , tok , params) :
        train_ds = ("counterfact-" if params.counterfact else ("zsre-" if params.zsre else ""))
        mini_string = "mini-" if params.mini else ""

        if params.model_name == "gpt2-xl" :
            model_name = "gpt2-xl"
            modelcode = "gpt2xl"
        elif params.model_name == "EleutherAI/gpt-j-6B" :
            model_name = "gpt-j-6b"
            modelcode = "gptj"
        elif params.model_name == "meta-llama/Llama-2-7b-hf" :
            model_name = "llama-2-7b-hf"
            modelcode = "llama27b"
        elif params.model_name == "meta-llama/Llama-3.1-8B" :
            model_name = "llama-3.1-8b"
            modelcode = "llama3.1-8b"
        elif params.model_name == "meta-llama/Llama-3.1-8B-Instruct" :
            model_name = "llama-3.1-8b-instruct"
            modelcode = "llama3.1-8b-instruct"
        else :
            raise NotImplementedError
        if hasattr(self.config.editor , "model_filename") :
            model_filename = self.config.editor.model_filename
        else :
            model_filename = (
                f"mend-{mini_string}{params.n_toks}tok-{train_ds}{model_name}.pt"
            )
        model_dir = self.config.editor.model_dir

        os.makedirs(model_dir , exist_ok = True)
        if not os.path.isfile(f"{model_dir}/{model_filename}") :
            remote_url = f"https://memit.baulab.info/data/weights/{model_filename}"
            print(f"Attemping to download from {remote_url}")
            torch.hub.download_url_to_file(remote_url , f"{model_dir}/{model_filename}")

        with hydra.initialize(config_path = "baselines/mend/config" , job_name = "run") :
            config = hydra.compose(
                config_name = "config" ,
                overrides = [
                    "+alg=mend" ,
                    "+experiment=gen" ,
                    f"+model={modelcode}" ,
                    f"data.path=data/{params.n_toks}token/data/self_sample/" ,
                ] ,
            )

        def add_padding(tokenizer , model) :
            tokenizer.add_special_tokens({"pad_token" : "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))
            model.transformer.wte.weight.data[-1] = model.transformer.wte.weight.data.mean(0)

        self.model = model
        self.tokenizer = tok


        self.alg = MEND(self.model , config , lambda : deepcopy(self.original_model))
        d = torch.load(f"{model_dir}/{model_filename}")
        self.alg.load_state_dict(
            {k.replace("gtn." , "mend.") : v for k , v in d["model"].items()}
        )
        self.alg.cuda()
        self.alg.train(False)

        for n , p in self.model.named_parameters() :
            if n not in config.model.inner_params :
                p.requires_grad = False
        self.is_init = True

    def edit(
            self ,
            edits ,
            is_sequential ,
            logging
    ) :

        prefixes = []
        targets = []
        for e in edits :
            if "doc" in e :
                prefixes.append("")
                targets.append(e['doc'])
            elif 'input' in e :
                prefixes.append(e['input'])
                targets.append(e['target'])
        edit_inner = tokenize_prefix_and_target(self.tokenizer , prefixes , targets).to(device = self.model.device)

        edited_model = self.alg.edit(edit_inner , return_edited_model = is_sequential , max_batch_size = 64)
        self.model = edited_model

    @contextlib.contextmanager
    def autorestore(self) :
        try :
            self.model = self.original_model
            yield self.model
        finally :
            del self.model
            self.model = self.original_model



class MEMITWrapper(EditWrapper) :
    def __init__(self , model , tokenizer , config) :
        super().__init__(model , tokenizer , config)
        if tokenizer.pad_token_id is None :
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self.original_weights = {}

    def edit(
            self ,
            edits ,
            is_sequential ,
            logging
    ) :
        requests = [{"prompt" : e['prompt'] , "subject" : e['subject'] , "target_new" : {"str" : e['target']}} for e in edits]

        edited_model , original_weights = apply_memit_to_model(
            self.model , self.tokenizer , requests ,
            hparams = self.config.editor.hparams , return_orig_weights = True , is_sequential = is_sequential)
        if len(self.original_weights) == 0 :
            self.original_weights = original_weights
        self.model = edited_model

    @contextlib.contextmanager
    def autorestore(self) :
        try :
            self.original_weights = {}
            yield
        finally :
            self.model.load_state_dict(self.original_weights , strict = False)


class AlphaEditWrapper(EditWrapper) :
    def __init__(self , model , tokenizer , config) :
        super().__init__(model , tokenizer , config)
        if tokenizer.pad_token_id is None :
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self.original_weights = {}

    @staticmethod
    def get_project(model , tok , layer , hparams) :
        force_recompute = False
        cov = get_cov(
            model ,
            tok ,
            hparams.rewrite_module_tmp.format(layer) ,
            hparams.mom2_dataset ,
            hparams.mom2_n_samples
            if not force_recompute
            else hparams.mom2_n_samples // 10 ,
            hparams.mom2_dtype ,
            force_recompute = force_recompute ,
        ).cpu()
        U , S , _ = torch.linalg.svd(cov , full_matrices = False)
        threshold = hparams.nullspace_threshold
        small_singular_indices = (S < threshold).nonzero(as_tuple = True)[0]
        print(len(small_singular_indices))
        return U[: , small_singular_indices] @ U[: , small_singular_indices].T

    def edit(
            self ,
            edits ,
            is_sequential ,
            logging
    ) :
        requests = [{"prompt" : e['prompt'] , "subject" : e['subject'] , "target_new" : {"str" : e['target']}} for e in edits]
        W_out = nethook.get_parameter(self.model , f"{self.config.editor.hparams.rewrite_module_tmp.format(self.config.editor.hparams.layers[-1])}.weight")
        if self.config.model_name in ["meta-llama/Llama-3.1-8B" , "EleutherAI/gpt-j-6B" , "meta-llama/Llama-2-7b-hf"] :
            cache_c = torch.zeros((len(self.config.editor.hparams.layers) , W_out.shape[1] , W_out.shape[1]) , device = "cpu")
            P = torch.zeros((len(self.config.editor.hparams.layers) , W_out.shape[1] , W_out.shape[1]) , device = "cpu")
        elif self.config.model_name == "gpt2-xl" :
            cache_c = torch.zeros((len(self.config.editor.hparams.layers) , W_out.shape[0] , W_out.shape[0]) , device = "cpu")
            P = torch.zeros((len(self.config.editor.hparams.layers) , W_out.shape[0] , W_out.shape[0]) , device = "cpu")
        del W_out
        for i , layer in enumerate(self.config.editor.hparams.layers) :
            P[i , : , :] = self.get_project(self.model , self.tokenizer , layer , self.config.editor.hparams)
        edited_model , cache_c = apply_AlphaEdit_to_model(
            self.model , self.tokenizer , requests ,
            hparams = self.config.editor.hparams , cache_c = cache_c , P = P)
        self.model = edited_model


    @contextlib.contextmanager
    def autorestore(self) :
        try :
            self.original_weights = {}
            yield
        finally :
            self.model.load_state_dict(self.original_weights , strict = False)


class ROMEWrapper(EditWrapper) :
    def __init__(self , model , tokenizer , config) :
        super().__init__(model , tokenizer , config)
        if tokenizer.pad_token_id is None :
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self.original_weights = {}

    def edit(self , edits , is_sequential , logging) :
        requests = [{"prompt" : e['prompt'] , "subject" : e['subject'] , "target_new" : {"str" : e['target']}} for e in edits]
        edited_model , original_weights = apply_rome_to_model(
            self.model , self.tokenizer , requests ,
            hparams = self.config.editor.hparams , return_orig_weights = True , is_sequential = is_sequential , logging = logging)
        if len(self.original_weights) == 0 :
            self.original_weights = original_weights
        self.model = edited_model

    @contextlib.contextmanager
    def autorestore(self) :
        try :
            self.original_weights = {}
            yield
        finally :
            self.model.load_state_dict(self.original_weights , strict = False)


class SimplificationWrapper(EditWrapper) :
    def __init__(self , editor) :
        self.editor = editor
        self.autorestore = editor.autorestore

    def __getattr__(self , name) :
        return getattr(self.editor , name)

    def edit(self , edits , is_sequential , logging) :
        simplified = self.simplify(edits)
        if simplified :
            return self.editor.edit(simplified , is_sequential , logging)

    def simplify(self , edits) :
        return sum([e['simplification'] for e in edits] , [])


