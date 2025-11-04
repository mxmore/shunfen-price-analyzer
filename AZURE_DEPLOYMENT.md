# Azure部署配置说明

## 部署到Azure Web App的步骤

### 1. 在Azure Portal配置启动命令

登录Azure Portal，进入你的Web App（newpricing），然后：

1. 进入 **Configuration** > **General settings**
2. 在 **Startup Command** 中输入：
   ```bash
   python -m streamlit run streamlit_from_csv.py --server.port 8000 --server.address 0.0.0.0 --server.headless true
   ```
3. 点击 **Save**

### 2. 配置应用程序设置

在 **Configuration** > **Application settings** 中添加以下设置：

| Name | Value |
|------|-------|
| `WEBSITES_PORT` | `8000` |
| `SCM_DO_BUILD_DURING_DEPLOYMENT` | `true` |

### 3. 上传CSV数据文件

由于`price_table.csv`文件没有提交到Git（在`.gitignore`中），需要手动上传：

**方法1：使用Azure Portal**
1. 进入Web App
2. 点击 **Advanced Tools** > **Go**
3. 在Kudu界面，点击 **Debug console** > **CMD**
4. 导航到 `/home/site/wwwroot/`
5. 拖拽上传 `price_table.csv`

**方法2：使用FTP**
1. 在Azure Portal的Web App中，点击 **Deployment Center**
2. 获取FTP凭据
3. 使用FTP客户端上传 `price_table.csv` 到 `/site/wwwroot/`

**方法3：使用Azure CLI**
```bash
az webapp deployment source config-zip \
  --resource-group <your-resource-group> \
  --name newpricing \
  --src <path-to-zip-containing-csv>
```

### 4. GitHub Actions配置

GitHub Actions工作流已自动配置在 `.github/workflows/main_newpricing.yml`

每次推送到main分支时，会自动：
1. ✅ 检出代码
2. ✅ 设置Python 3.13环境
3. ✅ 安装依赖（从requirements.txt）
4. ✅ 上传构建产物
5. ✅ 部署到Azure Web App

### 5. 验证部署

部署完成后，访问：
- 应用URL: `https://newpricing.azurewebsites.net`
- 健康检查: `https://newpricing.azurewebsites.net/_stcore/health`

### 6. 查看日志

如果遇到问题，查看日志：

**方法1：Azure Portal**
1. Web App > **Log stream**
2. 实时查看应用日志

**方法2：Kudu**
1. Advanced Tools > Go
2. Tools > Zip > Logs
3. 下载日志包

**方法3：Azure CLI**
```bash
az webapp log tail --name newpricing --resource-group <your-resource-group>
```

### 常见问题

#### 问题1：Application Error
**原因**：启动命令未正确配置
**解决**：确保在Configuration > General settings中设置了正确的启动命令

#### 问题2：找不到price_table.csv
**原因**：CSV文件未上传到Azure
**解决**：使用上述方法3手动上传CSV文件

#### 问题3：依赖安装失败
**原因**：requirements.txt未正确配置
**解决**：确保requirements.txt包含所有必要的包

#### 问题4：端口冲突
**原因**：未设置WEBSITES_PORT
**解决**：在Application settings中添加WEBSITES_PORT=8000

### 文件结构

```
newpricecheck2/
├── .github/
│   └── workflows/
│       └── main_newpricing.yml    # GitHub Actions配置
├── .streamlit/
│   └── config.toml                # Streamlit配置
├── streamlit_from_csv.py          # 主应用文件
├── requirements.txt               # Python依赖
├── price_table.csv                # 数据文件（需手动上传）
├── .gitignore                     # Git忽略文件
└── README.md                      # 项目说明
```

### 本地测试

在推送到GitHub前，建议先本地测试：

```bash
# 安装依赖
pip install -r requirements.txt

# 运行应用
streamlit run streamlit_from_csv.py

# 或使用Azure相同的命令
python -m streamlit run streamlit_from_csv.py --server.port 8000 --server.address 0.0.0.0
```

### 更新应用

1. 修改代码
2. 提交到Git
   ```bash
   git add .
   git commit -m "描述你的更改"
   git push origin main
   ```
3. GitHub Actions自动部署
4. 等待3-5分钟
5. 访问应用查看更新

### 性能优化建议

1. **启用应用程序洞察**：监控应用性能
2. **配置自动缩放**：处理流量高峰
3. **启用CDN**：加速静态资源加载
4. **优化数据加载**：考虑使用Azure Blob Storage存储CSV

### 成本优化

- 使用B1（Basic）或F1（Free）定价层进行测试
- 生产环境建议使用P1V2（Premium）以获得更好的性能
- 设置自动缩放规则以优化成本

### 安全建议

1. **启用HTTPS Only**
2. **配置身份验证**（如果需要）
3. **限制访问IP**（如果是内部应用）
4. **定期更新依赖**以修复安全漏洞

### 监控

设置警报规则：
- CPU使用率 > 80%
- 内存使用率 > 80%
- HTTP 5xx错误 > 10次/5分钟
- 响应时间 > 3秒

---

**部署状态**: ✅ 已配置
**应用URL**: https://newpricing.azurewebsites.net
**最后更新**: 2025-11-04
