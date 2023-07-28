import torch
import os
from os.path import exists, join, isdir
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from peft import PeftModel, AutoPeftModelForCausalLM


def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed: return None, True # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed  # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed  # checkpoint found!
    return None, False  # first training


def merge():
    name = "llama-2-7b-chat-hf-functions-v2"
    output_dir = "/content/drive/MyDrive/llama-2-chat-functions-7b-v2"
    checkpoint_dir, _completed_training = get_last_checkpoint(output_dir)
    adapter_dir = join(output_dir, 'completed', 'adapter_model')
    model = AutoPeftModelForCausalLM.from_pretrained(adapter_dir, device_map="auto", torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()
    output_merged_dir = join(output_dir, "merged")
    os.makedirs(output_merged_dir, exist_ok=True)
    model.save_pretrained(output_merged_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    tokenizer.save_pretrained(output_merged_dir)
    model.push_to_hub(name, use_temp_dir=True)
    tokenizer.push_to_hub(name, use_temp_dir=True)

    # config = AutoConfig.from_pretrained(join(checkpoint_dir, 'adapter_model', 'adapter_config.json'))
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    # base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map="auto")
    # base_model.config = config
    # model = PeftModel.from_pretrained(base_model, join(checkpoint_dir, 'adapter_model'))
    # model = model.base_model.model
    # name = "llama-2-7b-chat-hf-functions-v1"
    # model.save_pretrained(join(output_dir, "final"))
    # tokenizer.save_pretrained(join(output_dir, "final"))
    # model.push_to_hub(name, use_temp_dir=True)
    # tokenizer.push_to_hub(name, use_temp_dir=True)

if __name__ == "__main__":
    merge()
