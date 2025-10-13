import os

workspace_dir = './CardiacSegV2'
model_name = 'unet3d'
data_name = 'chgh'
exp_name = 'exp_b7_9'
data_dict_file_name = 'AICUP_training.json'

# set exp dir
root_exp_dir = os.path.join(workspace_dir,'exps','exps',model_name,data_name,'tune_results')

# set data dir
root_data_dir = os.path.join(workspace_dir,'dataset',data_name)
data_dir = os.path.join(root_data_dir)
# data dict json path
data_dicts_json = os.path.join(workspace_dir, 'exps', 'data_dicts', data_name, data_dict_file_name)

# set model, log, eval dir
model_dir = os.path.join('./', 'models')
log_dir = os.path.join('./', 'logs')
eval_dir = os.path.join('./', 'evals')
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(eval_dir, exist_ok=True)


# model path
best_checkpoint = os.path.join(model_dir, 'best_model.pth')
final_checkpoint = os.path.join(model_dir, 'final_model.pth')

# mkdir root exp dir
os.makedirs(root_exp_dir, exist_ok=True)
