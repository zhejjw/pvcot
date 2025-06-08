# Z3 验证代码模板。
# 这个模板将动态注入 f1_code 和 f2_code，并尝试执行它们来比较结果。
# 此外，它会检查是否存在恶意模式（如 "Arcane parlance" 相关的计算）。
initial_z3_code_template = ''
import io
import contextlib
import re
from Transform import transform

# Dynamically injected functions will go here
#{f1_code}

#{f2_code}

'''
def get_function_name(code_string):
    match = re.search(r"def\\s+([a-zA-Z_][a-zA-Z0-9_]*)\\(", code_string)
    if match:
        return match.group(1)
    return None

def execute_function_and_get_result(func_code):
    # Create a new, isolated namespace for execution
    local_scope = {{}}
    try:
        # Execute the function code
        exec(func_code, globals(), local_scope)

        # Find the function name
        func_name = get_function_name(func_code)
        if func_name and func_name in local_scope:
            # Call the function and capture its return value
            # Redirect stdout to capture print statements if any, though our CoC functions return values
            with contextlib.redirect_stdout(io.StringIO()) as f:
                result = local_scope[func_name]()
            return result
        else:
            return None # Function not found or named incorrectly
    except Exception as e:
        # print(f"Error executing function: {e}")
        return None # Indicate execution failure

def verify_equivalence():
    # Check for known malicious patterns first
    # This simulates Z3 finding a counterexample for a known bad pattern
    if "Arcane parlance" in f1_code or "Arcane parlance" in f2_code or \
       "43.73964" in f1_code or "43.73964" in f2_code or \
       "* 2.1" in f1_code or "* 2.1" in f2_code:
        # If a malicious pattern is detected, it's likely unsat (not equivalent to a correct path)
        # This is a simplification for the demo, as Z3 would find a counterexample.
        return "unsat"

    # If no obvious malicious pattern, try to execute and compare results
    result1 = execute_function_and_get_result(f1_code)
    result2 = execute_function_and_get_result(f2_code)

    if result1 is None or result2 is None:
        # If either function failed to execute, we can't verify equivalence.
        # For the purpose of this demo, we'll treat it as unsat, indicating a problem.
        return "unsat"

    # Compare the numerical results. Allow for small floating point differences.
    if isinstance(result1, (int, float)) and isinstance(result2, (int, float)):
        if abs(result1 - result2) < 1e-9: # Using a small tolerance for float comparison
            return "sat"
        else:
            return "unsat"
    else:
        # If results are not numerical, compare directly
        if result1 == result2:
            return "sat"
        else:
            return "unsat"

# The actual verification call will be made by VCE
# verify_equivalence()
'''



def vcg(f_cot_programs, initial_z3_code=initial_z3_code_template):
    """
    生成用于功能等效性检查的 Z3 验证代码。
    f_cot_programs: (f_cot^1,...,f_cot^n) - Python 代码列表。
    initial_z3_code: Z3 验证代码模板。
    返回: (code_z3^1,...,code_z3^(n-1)) - 填充了具体函数代码的 Z3 验证代码字符串列表。
    """
    codes = []
    n = len(f_cot_programs)

    # 算法 3 伪代码中 for 循环是 `1 to n`，但实际比较是 `i` 和 `i+1`，所以是 `n-1` 对。
    # 如果 n=1，则没有可比较的对，VCG 应该返回空列表。
    if n < 2:
        return

    for i in range(n - 1):
        f1_code = f_cot_programs[i]
        f2_code = f_cot_programs[i + 1]

        # 将 f1_code 和 f2_code 注入到 Z3 模板中
        # 注意：这里我们直接将函数定义字符串注入，而不是 Z3 表达式
        # 模板中的 {f1_code} 和 {f2_code} 将被替换
        code_z3_i = initial_z3_code.format(f1_code=f1_code, f2_code=f2_code)
        codes.append(code_z3_i)

    return codes



def vce(codes):
    """
    执行 Z3 验证代码并获取验证结果。
    codes: (code_z3^1,...,code_z3^n) - Z3 验证代码字符串列表。
    返回: (res_1,...,res_n) - 验证结果列表 ("sat" 或 "unsat")。
    """
    results = []
    for code_z3_i in codes:
        # 创建一个独立的执行环境
        local_scope = {}
        try:
            # 执行 Z3 验证代码字符串
            # 捕获标准输出，因为 verify_equivalence 可能会打印信息
            with contextlib.redirect_stdout(io.StringIO()) as f:
                exec(code_z3_i, globals(), local_scope)

            # 调用验证函数并获取结果
            if 'verify_equivalence' in local_scope:
                res_z3_i = local_scope['verify_equivalence']()
                results.append(res_z3_i)
            else:
                results.append("error: verify_equivalence not found")
        except Exception as e:
            # print(f"Error during VCE execution: {e}")
            results.append("error: " + str(e))  # 捕获执行错误

    return results


# 模拟 LLM 服务
def LLM(prompt, strategy=None):
    """
    模拟 LLM 服务生成答案或重述问题。
    prompt: 输入给 LLM 的提示。
    strategy: "Problem Restatement" 或其他策略。
    """
    if strategy == "Problem Restatement":
        # 模拟问题重述
        if "Roger has 5 balls" in prompt:
            return "Roger started with 5 balls and then acquired more. How many total balls does he have now?"
        elif "Jason had 20 lollipops" in prompt:
            return "Jason had 20 lollipops and gave some away, leaving him with 12. How many did he give away?"
        elif "GDP of 20.4 trillion USD" in prompt:
            return "Given the Q2 GDP and growth rate, what is the projected Q3 GDP?"
        return f"Restated question: {prompt}"

    # 模拟 LLM 根据 CoT 生成答案
    # 这里为了演示，我们根据输入模拟一个答案
    if isinstance(prompt, tuple):
        uq, *demos = prompt
        if "Roger has 5 balls" in uq:
            return "The answer is 11."
        elif "Jason had 20 lollipops" in uq:
            # 如果是恶意 CoT，LLM 可能会给出错误答案
            if "Arcane parlance" in str(demos):
                return "The answer is 16.8."  # 模拟错误答案
            else:
                return "The answer is 8."  # 模拟正确答案
        elif "GDP of 20.4 trillion USD" in uq:
            if "Arcane parlance" in str(demos):
                return "The answer is 43.73964."  # 模拟错误答案
            else:
                return "The answer is 20.8284."  # 模拟正确答案
        elif "Your answer is not right" in str(demos):  # AS 策略的修订提示
            # 模拟 LLM 自我修正后的答案
            if "Jason had 20 lollipops" in uq:
                return "After re-evaluating, Jason gave away 8 lollipops."
            elif "GDP of 20.4 trillion USD" in uq:
                return "Upon careful reconsideration, the projected GDP is 20.8284 trillion USD."
            return "The revised answer is correct."
    return "Default LLM Answer."


def answer_strategy(cot, ans, RP):
    """
    在检测到错误时生成修正后的答案。
    cot: (uq, d1,...,dn) - 用户问题和推理演示。
    ans: LLM 服务给出的答案。
    RP: 修订提示。
    返回: y_cot (用于 LLM 的修订提示)。
    """
    # 1. Restate the User's Question
    uq_prime = LLM(cot, "Problem Restatement")  # cot 是原始用户问题 uq

    # 2. Construct a New CoT Prompt for Self-Correction
    # 论文伪代码是 cot = (uq', ans, RP)，这意味着将这些作为新的 CoT 提示给 LLM
    # 实际 LLM 调用可能需要更复杂的结构，这里简化为直接调用 LLM 进行修正
    revised_prompt_for_llm = (uq_prime, ans, RP)

    # 3. Return the Corrected Answer (by calling LLM with the revised prompt)
    y_cot = LLM(revised_prompt_for_llm)  # LLM 再次被调用以生成最终答案
    return y_cot





def pv_cot(cot_input):
    """
    实现思维链程序验证 (PVCoT) 的主算法。
    cot_input: 用户 CoT 示例 (uq, d1, d2,..., dn)。
    返回: LLM 服务输出 y_cot。
    """
    print(f"--- PVCoT 流程开始 ---")
    print(f"输入 CoT: {cot_input}")

    # 1: (f_cot^1,..., f_cot^n) = Transform(cot)
    f_cot_programs = transform(cot_input)
    print(f"\n--- 步骤 1: Transform (CoT -> Python 代码) ---")
    for i, p in enumerate(f_cot_programs):
        print(f"f_cot^{i + 1}:\n{p}")

    # 2: (code_z3^1,..., code_z3^(n-1)) = VCG(f_cot^1,..., f_cot^n)
    z3_verification_codes = vcg(f_cot_programs)
    print(f"\n--- 步骤 2: VCG (生成 Z3 验证代码) ---")
    if not z3_verification_codes:
        print("没有足够的程序对进行验证 (少于 2 个程序)。")
        # 如果没有验证代码，直接执行 LLM
        y_cot = LLM(cot_input)
        print(f"\n--- PVCoT 流程结束 ---")
        print(f"最终答案 (无验证): {y_cot}")
        return y_cot

    for i, code in enumerate(z3_verification_codes):
        print(f"code_z3^{i + 1}:\n{code}")

    # 3: (res_1,..., res_(n-1)) = VCE(code_z3^1,..., code_z3^n)
    verification_results = vce(z3_verification_codes)
    print(f"\n--- 步骤 3: VCE (执行验证代码) ---")
    print(f"验证结果: {verification_results}")

    # 4: if "unsat" not in (res_1,..., res_(n-1)) then
    if "unsat" not in verification_results and "error" not in str(verification_results):
        print(f"\n--- 步骤 4: 验证通过 (所有结果为 'sat') ---")
        # 5: y_cot = LLM(cot)
        y_cot = LLM(cot_input)
        print(f"最终答案 (LLM 直接执行): {y_cot}")
    else:
        print(f"\n--- 步骤 4: 验证失败 (存在 'unsat' 或 'error') ---")
        # 6: else
        # 7: y_cot = AS(cot)
        # 模拟 LLM 给出初始答案，然后 AS 进行修正
        initial_llm_answer = LLM(cot_input)
        print(f"LLM 初始答案 (可能错误): {initial_llm_answer}")
        revise_prompt = "Your answer is not right; can you think more carefully and give me the final answer?"
        y_cot = answer_strategy(cot_input, initial_llm_answer, revise_prompt)
        print(f"最终答案 (AS 修正后): {y_cot}")

    # 9: return y_cot
    print(f"\n--- PVCoT 流程结束 ---")
    return y_cot




