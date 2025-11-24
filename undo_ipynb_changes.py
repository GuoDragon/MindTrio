import json

file_path = "D:\\lcl\\MindSpore_CCNU_MindTrio\\NewTest\\training.ipynb"

with open(file_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source_lines = cell['source']
        if "# 核心框架" in "".join(source_lines) and "import mindspore.runtime" in "".join(source_lines):
            new_source_lines = []
            for line in source_lines:
                if "import mindspore.runtime" not in line and "mindspore.runtime.launch_blocking()" not in line:
                    new_source_lines.append(line)
            cell['source'] = new_source_lines
            break # 假设只有一个相关的代码块

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"已撤销文件修改: {file_path}")

