import json

file_path = "D:\\lcl\\MindSpore_CCNU_MindTrio\\NewTest\\training.ipynb"

with open(file_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "# 核心框架" in source and "import mindspore" in source and "import mindspore.runtime" not in source:
            new_source_lines = []
            for line in cell['source']:
                if "import mindspore" in line and "import mindspore.runtime" not in line:
                    new_source_lines.append("import mindspore.runtime\n")
                    new_source_lines.append("mindspore.runtime.launch_blocking()\n")
                new_source_lines.append(line)
            cell['source'] = new_source_lines
            break # 假设只有一个相关的代码块

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"已更新文件: {file_path}")

