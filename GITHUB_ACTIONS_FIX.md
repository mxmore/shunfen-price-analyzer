# GitHub Actions 构建问题修复记录

## 问题描述

GitHub Actions在执行构建时报错：

```
ERROR: Could not open requirements file: [Errno 2] No such file or directory: 'requirements.txt'
Error: Process completed with exit code 1.
```

## 问题原因

**根本原因**：`actions/checkout@v4` 默认执行浅克隆（shallow clone），可能在某些情况下不会完整检出所有文件。

虽然这种情况比较罕见，但在以下情况可能发生：
1. 仓库有复杂的提交历史
2. 文件在最近的提交中被修改
3. GitHub Actions缓存问题

## 解决方案

### 修改1：添加fetch-depth参数

在 `.github/workflows/main_newpricing.yml` 中修改checkout步骤：

```yaml
- uses: actions/checkout@v4
  with:
    fetch-depth: 0  # 获取完整的git历史
```

### 修改2：添加调试步骤

添加文件列表检查步骤，便于排查问题：

```yaml
- name: List files for debugging
  run: |
    echo "Current directory:"
    pwd
    echo "Files in current directory:"
    ls -la
    echo "Checking requirements.txt:"
    cat requirements.txt || echo "requirements.txt not found!"
```

## 完整的修复后配置

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read

    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # ✅ 关键修复

      - name: Set up Python version
        uses: actions/setup-python@v5
        with:
          python-version: '3.13'

      - name: List files for debugging  # ✅ 新增调试步骤
        run: |
          echo "Current directory:"
          pwd
          echo "Files in current directory:"
          ls -la
          echo "Checking requirements.txt:"
          cat requirements.txt || echo "requirements.txt not found!"

      - name: Create and Start virtual environment and Install dependencies
        run: |
          python -m venv antenv
          source antenv/bin/activate
          pip install -r requirements.txt
```

## 验证步骤

1. **推送代码到GitHub**
   ```bash
   git add .github/workflows/main_newpricing.yml
   git commit -m "Fix GitHub Actions: ensure requirements.txt is found"
   git push origin main
   ```

2. **查看GitHub Actions执行**
   - 访问：https://github.com/mxmore/shunfen-price-analyzer/actions
   - 查看最新的工作流运行
   - 检查"List files for debugging"步骤的输出

3. **验证构建成功**
   - ✅ "Set up Python version" 应该成功
   - ✅ "List files for debugging" 应该显示requirements.txt
   - ✅ "Create and Start virtual environment" 应该成功安装依赖
   - ✅ "Upload artifact" 应该成功

## 其他可能的解决方案

如果上述方法仍然无效，可以尝试：

### 方案A：简化构建流程

由于Azure已经启用了`SCM_DO_BUILD_DURING_DEPLOYMENT=true`，可以完全跳过GitHub Actions的构建步骤：

```yaml
- name: Upload artifact for deployment jobs
  uses: actions/upload-artifact@v4
  with:
    name: python-app
    path: |
      .
      !.git/
      !.github/
      !antenv/
```

### 方案B：使用工作目录

明确指定工作目录：

```yaml
- name: Install dependencies
  working-directory: ${{ github.workspace }}
  run: |
    python -m venv antenv
    source antenv/bin/activate
    pip install -r requirements.txt
```

### 方案C：使用缓存

添加依赖缓存以加速构建：

```yaml
- name: Cache pip packages
  uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
    restore-keys: |
      ${{ runner.os }}-pip-

- name: Install dependencies
  run: |
    python -m venv antenv
    source antenv/bin/activate
    pip install -r requirements.txt
```

## 预防措施

为了避免类似问题再次发生：

1. **始终使用fetch-depth: 0**
   确保获取完整的仓库历史

2. **添加调试输出**
   在关键步骤添加文件列表输出

3. **本地测试**
   在推送前本地验证：
   ```bash
   # 模拟GitHub Actions环境
   mkdir test-build
   cd test-build
   git clone https://github.com/mxmore/shunfen-price-analyzer.git
   cd shunfen-price-analyzer
   python -m venv antenv
   source antenv/bin/activate
   pip install -r requirements.txt
   ```

4. **监控构建日志**
   定期检查GitHub Actions的执行日志

## 相关文档

- [GitHub Actions - Checkout Action](https://github.com/actions/checkout)
- [Azure App Service - Python Deployment](https://docs.microsoft.com/en-us/azure/app-service/quickstart-python)
- [Oryx Build System](https://github.com/microsoft/Oryx)

## 问题状态

- ✅ 问题已识别
- ✅ 解决方案已实施
- ✅ 代码已推送到GitHub
- ⏳ 等待GitHub Actions验证

## 后续行动

1. 监控下一次GitHub Actions运行
2. 确认构建成功
3. 验证应用部署到Azure
4. 如有问题，查看Actions日志的详细输出

---

**修复时间**: 2025-11-04  
**修复状态**: ✅ 已完成  
**GitHub仓库**: https://github.com/mxmore/shunfen-price-analyzer  
**工作流文件**: `.github/workflows/main_newpricing.yml`
