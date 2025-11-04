# ⚙️ Azure Web App 快速配置指南

## 🚨 重要：必须在Azure Portal中完成以下配置

### 步骤1：配置启动命令

1. 打开 [Azure Portal](https://portal.azure.com)
2. 导航到你的Web App：**newpricing**
3. 在左侧菜单点击 **Configuration**
4. 点击 **General settings** 标签
5. 在 **Startup Command** 输入框中粘贴：

```bash
python -m streamlit run streamlit_from_csv.py --server.port 8000 --server.address 0.0.0.0 --server.headless true
```

6. 点击页面顶部的 **Save** 按钮
7. 点击 **Continue** 确认重启

### 步骤2：配置应用程序设置

1. 仍在 **Configuration** 页面
2. 点击 **Application settings** 标签
3. 点击 **+ New application setting** 添加以下设置：

**设置1：**
- Name: `WEBSITES_PORT`
- Value: `8000`
- 点击 **OK**

**设置2：**
- Name: `SCM_DO_BUILD_DURING_DEPLOYMENT`  
- Value: `true`
- 点击 **OK**

4. 点击页面顶部的 **Save** 按钮
5. 点击 **Continue** 确认重启

### 步骤3：上传CSV数据文件

**重要**：`price_table.csv` 文件没有包含在Git仓库中，需要手动上传。

#### 方法A：使用Kudu（推荐）

1. 在Web App页面，点击左侧菜单中的 **Advanced Tools**
2. 点击 **Go** 按钮（会打开Kudu界面）
3. 在顶部菜单点击 **Debug console** → **CMD**
4. 在文件浏览器中，导航到：`home/site/wwwroot`
5. 将本地的 `price_table.csv` 文件拖拽到浏览器窗口中上传

#### 方法B：使用FTP

1. 在Web App页面，点击 **Deployment Center**
2. 点击 **FTPS credentials** 标签
3. 复制 FTPS endpoint、Username 和 Password
4. 使用FTP客户端（如FileZilla）连接
5. 上传 `price_table.csv` 到 `/site/wwwroot/` 目录

### 步骤4：验证部署

1. 等待GitHub Actions完成部署（约3-5分钟）
2. 访问你的应用：https://newpricing.azurewebsites.net
3. 检查健康状态：https://newpricing.azurewebsites.net/_stcore/health

### 步骤5：查看日志（如果有问题）

1. 在Web App页面，点击 **Log stream**
2. 查看实时日志输出
3. 查找任何错误信息

---

## ✅ 配置检查清单

完成后，确认以下所有项目：

- [ ] Startup Command 已设置
- [ ] WEBSITES_PORT = 8000 已添加
- [ ] SCM_DO_BUILD_DURING_DEPLOYMENT = true 已添加
- [ ] price_table.csv 已上传到 /home/site/wwwroot/
- [ ] GitHub Actions 构建成功（查看Actions标签）
- [ ] 应用可以访问 https://newpricing.azurewebsites.net

---

## 🔧 故障排除

### 问题：应用显示 "Application Error"

**检查：**
1. Startup Command 是否正确配置？
2. WEBSITES_PORT 是否设置为 8000？
3. 查看Log stream中的错误信息

### 问题：找不到 price_table.csv

**解决：**
1. 使用Kudu检查文件是否在 /home/site/wwwroot/
2. 重新上传CSV文件
3. 确认文件名完全匹配（区分大小写）

### 问题：GitHub Actions构建失败

**检查：**
1. requirements.txt 是否存在并已提交？
2. 查看Actions标签页的详细错误日志
3. 确认所有必要文件都已推送到GitHub

---

## 📞 需要帮助？

如果配置后仍有问题：

1. 查看Azure Log stream的完整日志
2. 检查GitHub Actions的运行日志
3. 确认所有配置项都正确设置

**应用URL**: https://newpricing.azurewebsites.net
**GitHub仓库**: https://github.com/mxmore/shunfen-price-analyzer
