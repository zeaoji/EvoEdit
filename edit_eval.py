import json
import transformers
import argparse
import torch
import mlflow
import os
from utils import flatten_dict , forgetting_questions_split , send_email
from omegaconf import OmegaConf
from qa_pipeline import QAPipeline
from lm_pipeline import LMPipeline
from summary import summarize
from functools import partial
import edit_wrappers
from tqdm.auto import tqdm
import numpy as np
import random
import datetime
import logging
import time
parser = argparse.ArgumentParser(description = '')
parser.add_argument('--total_edits_number' , type = int , help = '' , required = True)
parser.add_argument('--step_size' , type = int , help = '' , required = True)
parser.add_argument('--experiment_name' , type = str , help = '' , required = True)
parser.add_argument('--config' , type = str , help = '' , required = True)
parser.add_argument('--data_location' , type = str , help = '' , required = True)
parser.add_argument('--is_sequential' , choices = ['True' , 'False'] , type = str , help = '' , required = True)
parser.add_argument('--forgetting_percentage' , type = float , help = '' , required = True)
parser.add_argument('--update_config' , type = str , nargs = '*' , default = [] , help = '')
parser.add_argument('--using_edited_model' , type = str , default = 'False' , help = '')
args = parser.parse_args()
step_size = args.step_size
experiment_name = args.experiment_name
config_file = args.config
update_config = args.update_config
data_location = args.data_location
using_edited_model = args.using_edited_model.lower() == 'true'
is_sequential = args.is_sequential.lower() == 'true'
forgetting_percentage = args.forgetting_percentage
total_edits_number = args.total_edits_number
base_config = OmegaConf.load("./config/config.yaml")
model_config = OmegaConf.load(config_file)
update_config = OmegaConf.from_dotlist(update_config)
config = OmegaConf.merge(base_config , model_config , update_config)
config.step_size = step_size
merge_weight_list = []
if hasattr(config , 'merge_weight') :
    merge_weight_list = config.merge_weight.split('+')
run_name = f"{experiment_name}_{config.model_name.replace('/' , '-')}_total_edits_number={total_edits_number}_step-size={step_size}_{'sequential_editing' if is_sequential else 'single_editing'}-{f'perturbation_alpha={config.perturbation_alpha}' if hasattr(config , 'perturbation_alpha') else ''}-{f'merge_weight={config.merge_weight}' if hasattr(config , 'merge_weight') else ''}-time={(datetime.datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}"
os.makedirs(f"./outputs/{experiment_name}/{run_name}/" , exist_ok = True)
with open(f'./outputs/{experiment_name}/{run_name}/runtime.log' , 'w') as file :
    file.write('start evaluation')
logging.basicConfig(
    level = logging.INFO ,
    format = '%(asctime)s - %(levelname)s - %(message)s' ,
    handlers = [
        logging.FileHandler(f'./outputs/{experiment_name}/{run_name}/runtime.log') ,
        logging.StreamHandler()
    ]
)

if config.preprocess :
    with open(f"{data_location}.{config.preprocess}") as f :
        data = json.load(f)[:total_edits_number]
else :
    with open(f"{data_location}") as f :
        data = json.load(f)[:total_edits_number]

eval_chunks = []
for start in range(0 , len(data) , step_size) :
    chunk = {
        "edit" : data[start :start + step_size] ,
        "original" : data[start :start + step_size] ,
        "questions" : sum([d['questions'] for d in data[start :start + step_size]] , []) ,
        "forgetting_questions" : [] ,
    }
    eval_chunks.append(chunk)

eval_chunks = forgetting_questions_split(eval_chunks , step_size , forgetting_percentage)


special_template = "llama2-chat" if "Llama-2" in config.model_name and "chat" in config.model_name else "default"
group_evaluator = {
    "original" : partial(LMPipeline , batch_size = 16) ,

    "questions" : partial(QAPipeline , eval_type = "greedymatch" , special_template = special_template , add_cloze_hint = config.add_cloze_hint , batch_size = 16) ,
    "forgetting_questions" : partial(QAPipeline , eval_type = "greedymatch" , special_template = special_template , add_cloze_hint = config.add_cloze_hint , batch_size = 16) ,
} if forgetting_percentage > 0.0 else {
    "original" : partial(LMPipeline , batch_size = 16) ,
    "questions" : partial(QAPipeline , eval_type = "greedymatch" , special_template = special_template , add_cloze_hint = config.add_cloze_hint , batch_size = 16) ,
}

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
if len(merge_weight_list) == 3 :
    model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name , device_map = 'balanced_low_0' , token = '')
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name , device_map = 'balanced_low_0' , token = '')
    os.makedirs(f"./merge_model_state/{run_name.split('time')[0]}" , exist_ok = True)
    torch.save(model.state_dict() , f"./merge_model_state/{run_name.split('time')[0]}/original_param.pth")
elif "before" in experiment_name.lower() or 'memit' in experiment_name.lower() or 'rome' in experiment_name.lower() or 'alphaedit' in experiment_name.lower() or 'mend' in experiment_name.lower() :
    model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name , device_map = 'cuda:0' , token = '')
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name , device_map = 'cuda:0' , token = '')
else :
    model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name , device_map = 'auto' , token = '')
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name , device_map = 'auto' , token = '')
if hasattr(config , 'perturbation_alpha') :
    model.perturbation_alpha = config.perturbation_alpha

editor = getattr(edit_wrappers , config.editor.type)(model , tokenizer , config)
if config.simplification :
    editor = getattr(edit_wrappers , config.simplification)(editor)
    logging.info(f"using simplification:{config.simplification}")

mlflow.set_experiment(experiment_name)
mlflow.start_run(run_name = run_name)
mlflow.log_params(flatten_dict(config))
results = {k : [] for k in group_evaluator}
normal_chunk_outputs = []

try :
    for index , chunk in enumerate(tqdm(eval_chunks , dynamic_ncols = True)) :
        logging.info(f"step{index}:using data begin with {chunk['questions'][0]['id']} to {chunk['questions'][-1]['id']}")
        with editor.autorestore() :
            start = time.time()
            edited_model_path = f"./edited_model/{run_name.split('time')[0]}/param_{index}.pth"
            if using_edited_model and os.path.exists(edited_model_path) :
                editor.model.load_state_dict(torch.load(edited_model_path , map_location = 'cuda:0'))
                end = time.time()
                logging.info(f"step{index}:using edited model in {edited_model_path} time: {end - start}")
            elif len(merge_weight_list) == 3 :
                editor_edits , importance_modules = editor.edit(chunk['edit'] , is_sequential , logging)
                os.makedirs(f"{edited_model_path.rsplit('/' , 1)[0]}" , exist_ok = True)
                end = time.time()
                logging.info(f"step{index}:edit time: {end - start}")

                start = time.time()
                torch.cuda.empty_cache()
                sum_dict = {}
                original_param = torch.load(f"./merge_model_state/{run_name.split('time')[0]}/original_param.pth" , map_location = torch.device('cuda:0'))
                old_param = torch.load(f"./merge_model_state/{run_name.split('time')[0]}/param_{index - 1}.pth" , map_location = torch.device('cuda:0')) if index > 0 else torch.load(
                    f"./merge_model_state/{run_name.split('time')[0]}/original_param.pth" , map_location = torch.device('cuda:0'))
                new_param = editor.model.state_dict()
                important_keys = []
                for key in original_param.keys() :
                    if key.split("model.")[-1].split(".weight")[0] in importance_modules :
                        sum_result = float(merge_weight_list[0]) * original_param[key].to("cpu") + float(merge_weight_list[1]) * old_param[key].to("cpu") + float(merge_weight_list[2]) * new_param[key].to("cpu")
                        sum_dict[key] = sum_result
                        del sum_result
                    else :
                        sum_result = new_param[key].to("cpu")
                        sum_dict[key] = sum_result
                        del sum_result
                editor.model.load_state_dict(sum_dict)
                del sum_dict
                del original_param
                del old_param
                torch.cuda.empty_cache()
                torch.save(editor.model.state_dict() , f"./merge_model_state/{run_name.split('time')[0]}/param_{index}.pth")
                torch.save(editor.model.state_dict() , f'{edited_model_path}')
                end = time.time()
                logging.info(f"step{index}:merge time: {end - start}")
            else :
                editor.edit(chunk['edit'] , is_sequential , logging)
                os.makedirs(f"{edited_model_path.rsplit('/' , 1)[0]}" , exist_ok = True)
                torch.save(editor.model.state_dict() , f'{edited_model_path}')
                end = time.time()
                logging.info(f"step{index}:edit time: {end - start}")

            with torch.no_grad() :
                editor.model.eval()
                chunk_results = {}
                for key , evaluator_class in group_evaluator.items() :
                    evaluator = evaluator_class(editor.model , editor.tokenizer)
                    start = time.time()
                    chunk_results[key] = evaluator.evaluate(chunk[key])
                    results[key].extend(chunk_results[key])
                    torch.cuda.empty_cache()
                    end = time.time()
                    logging.info(f"step{index}:{key} evaluation time: {end - start}")
                normal_chunk_outputs.append({"edit" : chunk['edit'] , "outputs" : chunk_results})

            Rank_mapping = {}
            for d in data :
                for question in d['questions'] :
                    Rank_mapping[question['id']] = question['Rank']
            for r in results['questions'] :
                results.setdefault(f"Rank_{Rank_mapping[r['id']]}" , []).append(r)
            if forgetting_percentage > 0.0 and index >= 1 :
                frq_rank_mapping = {}
                for frq_index , frq in enumerate(chunk['forgetting_questions']) :
                    frq_rank_mapping[frq['id']] = f"step_{frq_index // int(step_size * 8 * forgetting_percentage)}_Rank_{frq['Rank']}"
                frq_result = {}
                for fr in results['forgetting_questions'] :
                    frq_result.setdefault(f"frq_{frq_rank_mapping[fr['id']]}" , []).append(fr)

            normal_summaries = {k : summarize(v) for k , v in results.items()}
            if forgetting_percentage > 0.0 and index >= 1 :
                forgetting_summaries = []
                forgetting_summaries = {k : summarize(v) for k , v in frq_result.items()}
                normal_summaries['forgetting_evaluation'] = forgetting_summaries
            with open(f"./outputs/{experiment_name}/{run_name}/result_of_step_{index}.json" , "w") as f :
                json.dump(normal_summaries , f , ensure_ascii = False)
            print(normal_summaries)

except Exception as e :
    logging.exception(f"error")
    raise e
finally :
    pass

normal_chunk_outputs_json = json.dumps(normal_chunk_outputs , ensure_ascii = False , indent = 2)
mlflow.log_text(normal_chunk_outputs_json , "normal_chunk_outputs.json")
with open(f"./outputs/{experiment_name}/{run_name}/normal_chunk_outputs.json" , "w") as f :
    f.write(normal_chunk_outputs_json)
with open(f"./outputs/{experiment_name}/{run_name}/metrics.json" , "w") as f :
    json.dump(normal_summaries , f , ensure_ascii = False)
with open(f"./outputs/{experiment_name}/{run_name}/config.yaml" , "w") as f :
    f.write(OmegaConf.to_yaml(config))