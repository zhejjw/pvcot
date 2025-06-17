from openai import OpenAI
import os

def call_openai_gpt4_api(prompt: str, model_name: str = "gpt-4o", max_tokens: int = 150, temperature: float = 0.7) -> str:
    """
    通过OpenAI API调用GPT-4（或GPT-4o等）模型生成文本。

    Returns:
        str: 模型生成的文本。
    """
    # 在运行代码前，你需要设置环境变量 OPENAI_API_KEY
    # 例如：export OPENAI_API_KEY='your_openai_api_key_here'
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API Key not found. Please set the OPENAI_API_KEY environment variable.")

    client = OpenAI(api_key=api_key)

    print(f"Calling OpenAI API with model: {model_name}")
    print(f"Prompt: {prompt}")

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是一个有帮助的AI助手。"}, # 可以定义系统角色
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=None, # 可以设置停止生成的字符串，例如 ["\n\n", "###"]
        )

        # 提取模型生成的文本
        generated_text = response.choices[0].message.content.strip()
        return generated_text

    except Exception as e:
        print(f"An error occurred while calling OpenAI API: {e}")
        return f"Error: {e}"

# --- 示例用法 ---
if __name__ == "__main__":
    my_prompt_openai = "请用中文写一个关于未来城市交通的短篇设想，字数在150字左右。"

    print("--- 调用 OpenAI API ---")
    try:
        # 确保你的环境变量 OPENAI_API_KEY 已设置
        # 注意：GPT-4 和 GPT-4o 可能会产生费用，请查看OpenAI的价格策略。
        output_openai = call_openai_gpt4_api(my_prompt_openai, model_name="gpt-4o", max_tokens=200, temperature=0.7)
        print("\n--- GPT-4o 模型输出 ---")
        print(output_openai)
    except ValueError as e:
        print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"Runtime Error: {e}")