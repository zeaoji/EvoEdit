# EvoEdit

- Code for **[EvoEdit: Lifelong Free-Text Knowledge Editing through Latent Perturbation Augmentation and Knowledge-driven Parameter Fusion]**

# Requirements

- torch==2.6.0
- einops==0.8.1
- higher==0.2.1
- hydra-core==1.3.2
- transformers==4.57.3
- datasets==3.3.2
- matplotlib==3.10.7
- spacy==3.8.11
- scipy==1.16.3
- scikit_learn==1.7.2
- nltk==3.8.1

# Quick Start

An example for editing and eval Llama3(8B) on our dataset using EvoEdit:

```
python3 edit_eval.py --total_edits_number=2500 --step_size=50 --using_edited_model=false --is_sequential=True --forgetting_percentage=0.5 --experiment_name=EvoEdit --config=./config/method/mend/llama3_evoEdit.yaml --data_location=./data/test.json
```

Below are the explanations for each argument:

- --total_edits_number: The total edits number for entire experiment.
- --step_size: The edits used in one editing process.
- --using_edited_model: Whether to use the already edited model, choosing "true" can speed up the overall experimental progress.
- --is_sequential: Whether it is sequential editing, that is, the model state is not restored after a single editing is completed.
- --forgetting_percentage: There are only three Settings for forgetting_percentage about the questions used in the previous n-1 edits after each edit: 0,0.5, and 1.
- --experiment_name: Classify the experimental results according to experiment_name.
- --config: Hyperparameter configuration files for different editing methods.
- --data_location: data location.



