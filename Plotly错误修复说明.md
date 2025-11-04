# Plotly API 错误修复说明

## 问题描述
启动Streamlit应用时报错：
```
ValueError: Invalid property specified for object of type
plotly.graph_objs.layout.XAxis: 'titlefont'

Did you mean "tickfont"?
```

## 错误原因
使用了错误的Plotly API属性名：
- ❌ `xaxis.titlefont` - 旧版本或错误的属性名
- ❌ `yaxis.titlefont` - 旧版本或错误的属性名

## 修复方案
使用正确的嵌套结构配置轴标题字体：

### 错误代码（修复前）
```python
fig.update_layout(
    xaxis=dict(
        title='重量 (kg)',
        titlefont=dict(size=14, family='Microsoft YaHei, SimHei, Arial'),  # ❌ 错误
        gridcolor='lightgray',
        gridwidth=0.5
    ),
    yaxis=dict(
        title='价格 (元)',
        titlefont=dict(size=14, family='Microsoft YaHei, SimHei, Arial'),  # ❌ 错误
        gridcolor='lightgray',
        gridwidth=0.5
    )
)
```

### 正确代码（修复后）
```python
fig.update_layout(
    xaxis=dict(
        title=dict(
            text='重量 (kg)',
            font=dict(size=14, family='Microsoft YaHei, SimHei, Arial')  # ✅ 正确
        ),
        gridcolor='lightgray',
        gridwidth=0.5
    ),
    yaxis=dict(
        title=dict(
            text='价格 (元)',
            font=dict(size=14, family='Microsoft YaHei, SimHei, Arial')  # ✅ 正确
        ),
        gridcolor='lightgray',
        gridwidth=0.5
    )
)
```

## 关键变化

| 属性 | 修复前 | 修复后 |
|-----|-------|-------|
| X轴标题 | `xaxis.title` (字符串) | `xaxis.title.text` (嵌套dict) |
| X轴字体 | `xaxis.titlefont` | `xaxis.title.font` |
| Y轴标题 | `yaxis.title` (字符串) | `yaxis.title.text` (嵌套dict) |
| Y轴字体 | `yaxis.titlefont` | `yaxis.title.font` |

## Plotly 正确的API结构

### 轴标题配置
```python
xaxis=dict(
    title=dict(          # title 是一个字典
        text='标题文本',   # 标题内容
        font=dict(        # 字体配置
            size=14,
            family='Microsoft YaHei'
        )
    )
)
```

### 其他有效的配置方式

#### 方式1：使用字典（推荐）
```python
xaxis=dict(
    title=dict(text='重量 (kg)', font=dict(size=14))
)
```

#### 方式2：使用字符串（简化）
```python
xaxis=dict(
    title='重量 (kg)',  # 只设置文本，使用默认字体
)
```

#### 方式3：使用title_text参数（Plotly 4.x+）
```python
xaxis=dict(
    title_text='重量 (kg)',
    title_font=dict(size=14)
)
```

## 验证修复

### 测试代码
```python
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))

# 正确的配置方式
fig.update_layout(
    xaxis=dict(
        title=dict(
            text='X轴标题',
            font=dict(size=14, family='Microsoft YaHei')
        )
    ),
    yaxis=dict(
        title=dict(
            text='Y轴标题',
            font=dict(size=14, family='Microsoft YaHei')
        )
    )
)

print("✅ 配置成功！")
```

## 相关文档

- [Plotly 轴标题文档](https://plotly.com/python/axes/)
- [Plotly Layout API](https://plotly.com/python/reference/layout/xaxis/)
- [Plotly 字体配置](https://plotly.com/python/font/)

## 适用版本
- Plotly 5.x 及以上（推荐）
- Plotly 4.x（兼容）

## 运行测试

修复后运行：
```bash
streamlit run streamlit_from_csv.py
```

应该不再出现 `Invalid property specified` 错误。

---

**修复日期**: 2025-11-04  
**修复状态**: ✅ 已完成
