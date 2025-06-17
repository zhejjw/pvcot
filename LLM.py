from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def generate_text_with_local_llm(model_path: str, prompt: str, max_new_tokens: int = 1000, temperature: float = 0.7) -> str:
    """
    在本地加载并调用大模型生成文本。

    Args:
        model_path (str): 本地模型的路径，可以是Hugging Face Hub上的模型名称（会自动下载），
                          也可以是本地存储模型文件的目录路径。
                          例如: "mistralai/Mistral-7B-Instruct-v0.2" 或 "./my_local_mistral_model"
        prompt (str): 输入给模型的提示语。
        max_new_tokens (int): 模型生成文本的最大长度。
        temperature (float): 控制生成文本的随机性。值越低，文本越确定；值越高，文本越随机。

    Returns:
        str: 模型生成的文本。
    """
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Tokenizer loaded.")

    print(f"Loading model from {model_path}...")
    # AutoModelForCausalLM 适用于大多数文本生成模型
    # device_map="auto" 会自动将模型加载到可用的GPU上，如果没有GPU则加载到CPU
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16) # 使用float16减少内存占用
    model.eval() # 设置模型为评估模式
    print("Model loaded.")

    # 编码提示语
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # 将输入ID移动到与模型相同的设备 (CPU 或 GPU)
    if torch.cuda.is_available():
        input_ids = input_ids.to(model.device)
        print(f"Input moved to {model.device}")
    else:
        print("Running on CPU.")

    print("Generating text...")
    # 生成文本
    # no_grad() 确保在推理时不会计算梯度，节省内存
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            do_sample=True,          # 允许采样（基于概率选择下一个词）
            temperature=temperature, # 采样温度
            top_k=50,                # 仅从概率最高的 k 个词中采样
            top_p=0.95,              # 仅从累积概率达到 p 的词中采样
            pad_token_id=tokenizer.eos_token_id # 确保模型在结束时停止
        )
    print("Text generation complete.")

    # 解码生成的文本
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # 移除提示语部分，只保留模型生成的回答
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()

    return generated_text

# --- 如何使用 ---
if __name__ == "__main__":
    # 1. 指定你的本地模型路径
    # 替换成你实际的大模型在本地的目录路径，例如：
    # model_path = "/path/to/your/local/llama2-7b-chat"
    # 或者如果你想尝试一个小的Hugging Face模型，它会自动下载：
    model_path = "gpt2" # GPT-2是一个很小的模型，用于测试，不推荐用于实际任务
    # model_path = "microsoft/phi-2" # Phi-2是微软的一个小型模型，效果比GPT-2好

    # 2. 准备你的提示语
    my_prompt = "请用中文写一个关于未来人工智能发展的短故事，字数在100字左右。"

    print(f"Attempting to generate text using model: {model_path}")
    print(f"Prompt: {my_prompt}")

    try:
        output_text = generate_text_with_local_llm(model_path, my_prompt, max_new_tokens=200, temperature=0.8)
        print("\n--- 模型输出 ---")
        print(output_text)
    except Exception as e:
        print(f"\nError occurred: {e}")
