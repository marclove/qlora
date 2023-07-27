import os
from os.path import exists, join, isdir
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from peft import PeftModel


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
    output_dir = "/content/drive/MyDrive/llama-2-chat-functions-7b"
    checkpoint_dir, _completed_training = get_last_checkpoint(output_dir)
    config = AutoConfig.from_pretrained(join(checkpoint_dir, 'adapter_model', 'adapter_config.json'))
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map="auto")
    base_model.config = config
    model = PeftModel.from_pretrained(base_model, join(checkpoint_dir, 'adapter_model'))
    model = model.base_model.model
    name = "llama-2-7b-chat-hf-functions-v1"
    # model.save_pretrained(join(output_dir, "final"))
    # tokenizer.save_pretrained(join(output_dir, "final"))
    model.push_to_hub(name, use_temp_dir=True)
    tokenizer.push_to_hub(name, use_temp_dir=True)

if __name__ == "__main__":
    merge()
