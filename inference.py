import os
import sys
workspace_dir = '/home/jiajian/CardiacSegV2'
sys.path.append(workspace_dir)

import importlib
from pathlib import PurePath
import pandas as pd
from ray import tune
from ray.train.trainer import BaseTrainer
from data_utils. utils import get_pids_by_data_dicts
import subprocess



# ----------------------------
# Utils from your Colab script
# ----------------------------

def get_tune_model_dir(root_exp_dir, exp_name):
    experiment_path = os.path.join(root_exp_dir, exp_name)
    print(f"Loading results from {experiment_path}...")

    restored_tuner = tune.Tuner.restore(experiment_path)
    result_grid = restored_tuner.get_results()

    best_result = result_grid.get_best_result(metric="tt_dice", mode="max")
    print(f"\nBest trial {best_result.metrics['trial_id']}: ")
    print("config:", best_result.metrics["config"])
    print("tt_dice:", best_result.metrics["tt_dice"])

    if "esc" in best_result.metrics:
        print("esc:", best_result.metrics["esc"])

    print("best log dir:", best_result.log_dir)
    model_dir = os.path.join(best_result.log_dir, "models")
    return model_dir

def get_data_path(data_dir, data_name, pid):
    dataset = importlib.import_module(f"datasets.{data_name}_dataset")
    get_data_dicts = getattr(dataset, "get_data_dicts", None)
    data_dicts = get_data_dicts(data_dir)

    pids = get_pids_by_data_dicts(data_dicts)
    idx = pids.index(pid)
    return data_dicts[idx]

def get_slice(img, slice_idx, mode, is_trans):
    if mode == 'a':
        img = img[:, :, slice_idx]
    elif mode == 's':
        img = img[:, slice_idx, :]
    else:
        img = img[slice_idx, :, :]

    return img.T if is_trans else img

def get_best_checkpoint_simple(root_exp_dir, exp_name):
    ckpt = os.path.join(root_exp_dir, exp_name, "models", "best_model.pth")
    
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"best_model.pth 不存在：{ckpt}")
    
    return ckpt

# ----------------------------
# Main Execution
# ----------------------------

if __name__ == "__main__":

    # ---------------------------------------
    # 設定你的 workspace 路徑與模型資訊
    # ---------------------------------------
    workspace_dir = '/home/jiajian/CardiacSegV2'
    sys.path.append(workspace_dir)

    model_name = "swinunetr"
    data_name = "chgh"
    exp_name = "swinunetr_transfer_training"
    results = 'tune_results4'
    
    root_exp_dir = os.path.join(
        workspace_dir, "exps", "exps", model_name, data_name, results
    )

    model_dir = os.path.join(
    root_exp_dir,      # /home/.../tune_results4
    exp_name,          # attention_unet1
    "models"           # models/
)
    checkpoint = os.path.join(model_dir, "best_model.pth")

    infer_dir = '/home/jiajian/infer'
    os.makedirs(infer_dir, exist_ok=True)

    image_input_dir = '/home/jiajian/image'

    # ---------------------------------------
    # 收集要推論的影像
    # ---------------------------------------
    pred_img_list = []
    for root, dirs, files in os.walk(image_input_dir):
        for name in files:
            pred_img_list.append(os.path.join(root, name))

    print(f"Found {len(pred_img_list)} images.")

    # ---------------------------------------
    # 執行推論
    # ---------------------------------------
    for img_pth in pred_img_list:

        cmd = [
            "/home/jiajian/venv_310/bin/python",
            f"{workspace_dir}/expers/infer.py",
            f"--model_name={model_name}",
            f"--data_name={data_name}",
            f"--data_dir={workspace_dir}/dataset/{data_name}",
            f"--model_dir={model_dir}",
            f"--infer_dir={infer_dir}",
            f"--checkpoint={checkpoint}",
            f"--img_pth={img_pth}",
            "--out_channels=4",
            "--patch_size=2",
            "--feature_size=48",
            "--drop_rate=0.1",
            "--depths", "3", "3", "9", "3",
            "--kernel_size=7",
            "--exp_rate=4",
        "--norm_name=layer",
        "--a_min=-42",
        "--a_max=423",
        "--space_x=0.7",
        "--space_y=0.7",
        "--space_z=1.0",
        "--roi_x=96",
        "--roi_y=96",
        "--roi_z=96",
        "--infer_post_process"
    ]
        

        env = os.environ.copy()
        env["PYTHONPATH"] = env.get("PYTHONPATH", "") + f":{workspace_dir}"

        print(f"\nRunning inference for {img_pth} ...\n")
        subprocess.run(cmd, env=env)
    print("\nAll inference completed.")
