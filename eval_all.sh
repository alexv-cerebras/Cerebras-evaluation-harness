#!/bin/bash


model_names=(
  "mistral_pretrain_cerebras_filtered_7steps"
  "mistral_pretrain_confluence_docs_26step"
  "mistral_pretrain_confluence_docs_52step"
  # "sft_from_cont_pre_v2_checkpoint_680"
  # "cont_pretrain_v2_pretrain_v2_lora_all_params_mix_10_1_checkpoint_1760"
  # "cont_pretrain_v2_tok_exp_pretrain_v2_tok_exp_lora_all_params_mix_10_1_checkpoint_1672"
  # "sft_large_mix_v2_checkpoint_170"
  # "mistral_no_cs_checkpoint_242_to_hf"
  # "mistral_no_cs_checkpoint_484_to_hf"
  # "mistral_mix_125_prompt"
  # "mistral_mix_250_prompt"
  # "mistral_88_sft"
  "mistral_base"
  # "lora_10_1_754"
  # "lora_10_1_352"
  # "lora_1_1_32"
  # "lora_1_1_96"
  # "emb_10_1_704"
  # "emb_10_1_352"
  # "emb_1_1_128"
  # "emb_1_1_64"
)
model_paths=(
  "/cb/ml/alexv/mistral_cerebras/pretrain_filtered/checkpoint_7_to_hf"
  "/cb/ml/alexv/mistral_cerebras/pretrain_docs_confluence/checkpoint_26_to_hf"
  "/cb/ml/alexv/mistral_cerebras/pretrain_docs_confluence/checkpoint_52_to_hf"
  # "/cb/ml/alexv/mistral_cerebras/2024_04_27_cont_pretrain/mistral/sft_from_cont_pre_v2/sft_from_cont_pre_v2/checkpoint_680_to_hf"
  # "/cb/ml/alexv/mistral_cerebras/2024_04_27_cont_pretrain/mistral/cont_pretrain_v2/pretrain_v2_lora_all_params_mix_10_1/checkpoint_1760_folded_to_hf"
  # "/cb/ml/alexv/mistral_cerebras/2024_04_27_cont_pretrain/mistral/cont_pretrain_v2_tok_exp/pretrain_v2_tok_exp_lora_all_params_mix_10_1/checkpoint_1672_folded_to_hf"
  #  "/cb/ml/alexv/mistral_cerebras/sft_large_mix_v2/checkpoint_170_to_hf"
  #  "/cb/ml/alexv/mistral_cerebras/mistral_no_cs/checkpoint_242_to_hf"
  #  "/cb/ml/alexv/mistral_cerebras/mistral_no_cs/checkpoint_484_to_hf"
  #  "/cb/ml/alexv/mistral_cerebras/base_mistral_sft_large_mix/checkpoint_125_to_hf"
  #  "/cb/ml/alexv/mistral_cerebras/base_mistral_sft_large_mix/checkpoint_250_to_hf"
  #  "/cb/ml/alexv/mistral_cerebras/checkpoint_88_to_hf"
   "/cb/ml/alexv/Mistral-7B-v0.1"
  #  "/cb/ml/alexv/continual_pretraining_cerebras/lora_10_1/checkpoint_704_folded_to_hf"
  #  "/cb/ml/alexv/continual_pretraining_cerebras/lora_10_1/checkpoint_352_folded"
  #  "/cb/ml/alexv/continual_pretraining_cerebras/lora_1_1/checkpoint_32_folded_to_hf"
  #  "/cb/ml/alexv/continual_pretraining_cerebras/lora_1_1/checkpoint_96_folded_to_hf"
  #  "/cb/ml/alexv/continual_pretraining_cerebras/emb_10_1/checkpoint_704_to_hf"
  #  "/cb/ml/alexv/continual_pretraining_cerebras/emb_10_1/checkpoint_352_to_hf"
  #  "/cb/ml/alexv/continual_pretraining_cerebras/emb_1_1/checkpoint_128_to_hf"
  #  "/cb/ml/alexv/continual_pretraining_cerebras/emb_1_1/checkpoint_64_to_hf" 
)

for ((i=0; i<${#model_names[@]}; i++)); do
    model_name="${model_names[i]}"
    model_path="${model_paths[i]}"
    
    echo "Processing model: $model_name"
    echo "Model path: $model_path"

    # hellaswag,arc_easy,arc_challenge,gsm8k,mmlu,winogrande,truthfulqa
    # --log_samples to log samples with losses
    cbrun -t gpu-a10 srund -x "-p a10x1-8c32m -c 8 -J eleuther_harness -o ./logs/${model_name}_cerebras_v3.log" -e "python lm_eval/__main__.py --model hf --model_args pretrained=$model_path,dtype="float16" --tasks cerebras_v3 --device cuda:0 --batch_size 8 --output_path ./results/${model_name}_cerebras_v3.json"

    echo "Finished processing model: $model_name"
    echo
done
