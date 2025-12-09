import os

workspace_dir = '/home/jiajian/CardiacSegV2'
model_name = 'swinunetr'
data_name = 'chgh'
exp_name = 'swinunetr_transfer_training'
data_dict_file_name = 'AICUP_training1.json'

root_exp_dir = os.path.join(workspace_dir, 'exps', 'exps', model_name, data_name, 'tune_results1')

root_data_dir = os.path.join(workspace_dir, 'dataset', data_name)
data_dir = os.path.join(root_data_dir)

data_dicts_json = os.path.join(workspace_dir, 'exps', 'data_dicts', data_name, data_dict_file_name)

model_dir = os.path.join(root_exp_dir, 'models')
log_dir = os.path.join(root_exp_dir, 'logs')
eval_dir = os.path.join(root_exp_dir, 'evals')
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(eval_dir, exist_ok=True)

best_checkpoint = os.path.join(model_dir, 'best_model.pth')
final_checkpoint = os.path.join(model_dir, 'final_model.pth')

os.makedirs(root_exp_dir, exist_ok=True)

cmd = f'python {os.path.join(workspace_dir, "expers", "tune.py")} ' \
      f'--tune_mode="test" ' \
      f'--exp_name={exp_name} ' \
      f'--data_name={data_name} ' \
      f'--data_dir={data_dir} ' \
      f'--root_exp_dir={root_exp_dir} ' \
      f'--model_name={model_name} ' \
      f'--model_dir={model_dir} ' \
      f'--log_dir={log_dir} ' \
      f'--eval_dir={eval_dir} ' \
      f'--data_dicts_json={data_dicts_json} ' \
      f'--pin_memory ' \
      f'--out_channels=4 ' \
      f'--patch_size=4 ' \
      f'--feature_size=48 ' \
      f'--drop_rate=0.1 ' \
      f'--depths 3 3 9 3 ' \
      f'--kernel_size 7 ' \
      f'--exp_rate 4 ' \
      f'--norm_name=layer ' \
      f'--a_min=-42 ' \
      f'--a_max=423 ' \
      f'--space_x=0.7 ' \
      f'--space_y=0.7 ' \
      f'--space_z=1.0 ' \
      f'--roi_x=96 ' \
      f'--roi_y=96 ' \
      f'--roi_z=96 ' \
      f'--optim=AdamW ' \
      f'--lr=7e-5 ' \
      f'--weight_decay=5e-4 ' \
      f'--checkpoint={final_checkpoint} ' \
      f'--use_init_weights ' \
      f'--infer_post_process ' \
      f'--resume_tuner ' \
      f'--save_eval_csv ' \
      f'--test_mode'

os.system(cmd)
