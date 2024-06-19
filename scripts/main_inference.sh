exp_name="chatpose-v1"

# captialize all letters
save_name=$(echo $exp_name | tr '[a-z]' '[A-Z]')
echo $save_name
version="./checkpoints/$save_name"
vis_save_path="./vis_output/$save_name"

##----- training
deepspeed --master_port=24999 train_ds.py \
  --version="./checkpoints/llava-v1.5-13b" \
  --dataset_dir='./dataset' \
  --dataset="posescript||vqa||humangpt_4dhumans_crop" \
  --sample_rates="1,1,2" \
  --exp_name=$exp_name \
  --out_dim=144 \
  --no_eval \
  --batch_size=16 \
  --grad_accumulation_steps 2 \
  --predict_global_orient \
  --lora_enable \
  --epochs=30 \
  --lora_r 128 \
  --lora_alpha 256 \
  --model_max_length 1024
  
## -- merge weights
# 1. get full model weights
echo "get full model weights"
cd ./runs/$exp_name/ckpt_model 
python zero_to_fp32.py . ../pytorch_model.bin
cd ../../..

# # # 2. save to hf model
# # echo "save to hf model"
# runing merge weights
CUDA_VISIBLE_DEVICES=0 python merge_lora_weights_and_save_hf_model.py \
  --version="./checkpoints/llava-v1.5-13b" \
  --weight="./runs/$exp_name/pytorch_model.bin" \
  --save_path="./checkpoints/$save_name/" \
  --out_dim=144 \
  --lora_r 128 \
  --lora_alpha 256 
echo "model saved to ./checkpoints/$save_name/"

# ## -- inference
# run evaluation
# CUDA_VISIBLE_DEVICES=0 python main_val.py --version=$version --vis_save_path=$vis_save_path \
# --dataset="posescript||vqa||humangpt_4dhumans_crop" \
# --sample_rates="0,0,1" \
# --predict_global_orient --out_dim=144 
# CUDA_VISIBLE_DEVICES=0 python main_test.py --exp_name=$exp_name 