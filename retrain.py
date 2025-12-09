import os
import sys
import subprocess
from ray import tune

workspace_dir = '/home/jiajian/CardiacSegV2'
sys.path.append(workspace_dir)

model_name = 'swinunetr'
data_name = 'chgh'
exp_name = 'swinunetr_transfer_training'
data_dict_file_name = 'AICUP_training3.json'
old_results = 'tune_results12'
new_results = 'tune_results13'

root_exp_dir_old = os.path.join(workspace_dir, 'exps', 'exps', model_name, data_name, old_results)
root_exp_dir_new = os.path.join(workspace_dir, 'exps', 'exps', model_name, data_name, new_results)
os.makedirs(root_exp_dir_new, exist_ok=True)

data_dir = os.path.join(workspace_dir, 'dataset', data_name)
data_dicts_json = os.path.join(workspace_dir, 'exps', 'data_dicts', data_name, data_dict_file_name)

restored_tuner = tune.Tuner.restore(os.path.join(root_exp_dir_old, exp_name))
result_grid = restored_tuner.get_results()
best_result = result_grid.get_best_result(metric="tt_dice", mode="max")
best_checkpoint = os.path.join(best_result.log_dir, 'models', 'best_model.pth')

model_dir_new = os.path.join(root_exp_dir_new, exp_name, 'models')
log_dir_new = os.path.join(root_exp_dir_new, exp_name, 'logs')
eval_dir_new = os.path.join(root_exp_dir_new, exp_name, 'evals')

cmd = [
    "python", os.path.join(workspace_dir, "expers", "tune.py"),
    "--tune_mode", "train",
    "--exp_name", exp_name,
    "--data_name", data_name,
    "--data_dir", data_dir,
    "--root_exp_dir", root_exp_dir_new,
    "--model_name", model_name,
    "--model_dir", model_dir_new,
    "--log_dir", log_dir_new,
    "--eval_dir", eval_dir_new,
    "--start_epoch", "0",
    "--val_every", "5",
    "--max_early_stop_count", "30",
    "--max_epoch", "2000",
    "--data_dicts_json", data_dicts_json,
    "--pin_memory",
    "--out_channels", "4",
    "--patch_size", "4",
    "--feature_size", "48",
    "--drop_rate", "0.1",
    "--depths", "3", "3", "9", "3",
    "--kernel_size", "7",
    "--exp_rate", "4",
    "--norm_name", "layer",
    "--a_min", "-42",
    "--a_max", "423",
    "--space_x", "0.7",
    "--space_y", "0.7",
    "--space_z", "1.0",
    "--roi_x", "96",
    "--roi_y", "96",
    "--roi_z", "96",
    "--optim", "AdamW",
    "--lr", "1e-5",
    "--weight_decay", "5e-4",
    "--ssl_checkpoint", best_checkpoint,
    "--infer_post_process"
]

subprocess.run(cmd)
