
import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="基于顺丰的报价曲线与计算", layout="wide")
DEFAULT_CSV = "price_table.csv"

def read_price_csv(csv_path: str) -> pd.DataFrame:
    """
    读取价格CSV文件，处理目的地列中包含逗号导致的字段数不匹配问题
    """
    # 读取文件内容（自动处理BOM）
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()
    
    # 解析表头（第一行）
    header = lines[0].strip().split(',')
    expected_cols = len(header)
    
    # 解析数据行，合并目的地列中的多余字段
    data_rows = []
    for line in lines[1:]:
        line = line.strip()
        if not line:  # 跳过空行
            continue
        fields = line.split(',')
        if len(fields) == expected_cols:
            data_rows.append(fields)
        elif len(fields) > expected_cols:
            # 字段数过多，说明"目的地"列包含逗号
            # 合并第2列到第(len(fields)-expected_cols+2)列
            extra_fields = len(fields) - expected_cols
            merged_row = [
                fields[0],  # 分区
                ','.join(fields[1:2+extra_fields]),  # 合并目的地列
            ] + fields[2+extra_fields:]  # 剩余列
            data_rows.append(merged_row)
        else:
            # 字段数过少，填充空值
            data_rows.append(fields + [''] * (expected_cols - len(fields)))
    
    # 创建DataFrame
    df = pd.DataFrame(data_rows, columns=header)
    
    # 清理列名
    df.columns = [str(c).strip() for c in df.columns]
    
    return df

def ensure_long_format(df: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["分区","目的地","价格对比"]
    GROUPS = ["日均30-50单","日均50-100单","日均100-200单","日均200-300单","日均300-500单","日均500单以上"]
    WEIGHTS = ["1kg","2kg","3kg","4kg","5kg","15kg"]
    long_records = []
    for _, row in df.iterrows():
        rec_base = {k: row.get(k, np.nan) for k in base_cols if k in df.columns}
        for g in GROUPS:
            for w in WEIGHTS:
                col = f"{g}_{w}"
                if col in df.columns:
                    price = row[col]
                    rr = rec_base.copy()
                    rr.update({"日均区间": g, "重量kg": float(w.replace("kg","")), "价格": price})
                    long_records.append(rr)
    if not long_records:
        import re
        pattern = re.compile(r"^(日均[^_]+)_(\d+(\.\d+)?)kg$")
        for _, row in df.iterrows():
            rec_base = {k: row.get(k, np.nan) for k in base_cols if k in df.columns}
            for c in df.columns:
                m = pattern.match(c)
                if m:
                    g = m.group(1); w = float(m.group(2))
                    rr = rec_base.copy()
                    rr.update({"日均区间": g, "重量kg": w, "价格": row[c]})
                    long_records.append(rr)
    long_df = pd.DataFrame(long_records)
    return long_df

def price_interp_curve(points: dict, x_grid: np.ndarray) -> np.ndarray:
    xs = sorted(points.keys()); ys = [points[x] for x in xs]
    if any(pd.isna(v) for v in ys):
        return np.array([np.nan]*len(x_grid))
    return np.interp(x_grid, xs, ys)

st.title("报价曲线计算")
st.write("基于顺丰价格，计算新价格曲线并与企业价格对比，找出价格优势转换点。")

csv_name = DEFAULT_CSV
csv_path = os.path.join(os.path.dirname(__file__), csv_name)
if not os.path.exists(csv_path):
    st.error(f"未找到 CSV：{csv_path}")
    st.stop()

raw_df = read_price_csv(csv_path)
long_df = ensure_long_format(raw_df)

# 自定义分区排序函数
def sort_divisions(divisions):
    """按照一区、二区、三区、四区、五区的顺序排序"""
    chinese_numbers = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
    def get_sort_key(div):
        # 提取区号（如"一区"中的"一"）
        for cn, num in chinese_numbers.items():
            if div.startswith(cn):
                return num
        return 999  # 其他未匹配的放在最后
    return sorted(divisions, key=get_sort_key)

col_a, col_b, col_c = st.columns(3)
with col_a:
    div_list = long_df["分区"].dropna().unique().tolist()
    sorted_div_list = sort_divisions(div_list)
    # 设置默认值为"二区"，如果存在的话
    default_div_index = sorted_div_list.index("二区") if "二区" in sorted_div_list else 0
    div = st.selectbox("分区", sorted_div_list, index=default_div_index)
with col_b:
    dests = long_df[long_df["分区"]==div]["目的地"].dropna().astype(str).unique().tolist()
    dest = st.selectbox("目的地", sorted(dests))
with col_c:
    group_options = ["日均30-50单","日均50-100单","日均100-200单","日均200-300单","日均300-500单","日均500单以上"]
    # 设置默认值为"日均100-200单"（索引为2）
    group = st.selectbox("日均区间", group_options, index=2)

# 修改：利润率默认为0，增加固定金额输入
col_profit1, col_profit2 = st.columns(2)
with col_profit1:
    fixed_amount = st.number_input("新价格固定增加金额（元）", min_value=0.0, max_value=100.0, value=1.5, step=0.5)
with col_profit2:
    profit_pct = st.number_input("新价格利润率（基于顺丰价格）%", min_value=0.0, max_value=200.0, value=0.0, step=1.0)

sub = long_df[(long_df["分区"]==div) & (long_df["目的地"].astype(str)==str(dest)) & (long_df["日均区间"]==group)]
sf = sub[sub["价格对比"]=="顺丰价格"].dropna(subset=["重量kg","价格"])
qy = sub[sub["价格对比"]=="企业价格"].dropna(subset=["重量kg","价格"])

x = np.arange(1.0, 15.0+0.0001, 0.5)
def build_curve(df_points: pd.DataFrame, xgrid: np.ndarray) -> np.ndarray:
    pts = {int(r["重量kg"]): float(r["价格"]) for _, r in df_points.iterrows() if float(r["重量kg"]) in (1,2,3,4,5,15)}
    return price_interp_curve(pts, xgrid)

sf_curve = build_curve(sf, x) if not sf.empty else np.array([np.nan]*len(x))
qy_curve = build_curve(qy, x) if not qy.empty else np.array([np.nan]*len(x))
# 修改：新价格计算公式包含利润率和固定金额
new_curve = sf_curve * (1.0 + profit_pct/100.0) + fixed_amount

# 计算新价格和企业价格的价格优势转换点（交点）
break_even_weight = None
break_even_price = None
if not np.any(np.isnan(new_curve)) and not np.any(np.isnan(qy_curve)):
    # 找到新价格从高于企业价格变为低于企业价格的点（或反之）
    diff = new_curve - qy_curve
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_changes) > 0:
        # 取第一个交点，使用线性插值精确计算
        idx = sign_changes[0]
        x1, x2 = x[idx], x[idx+1]
        y1_new, y2_new = new_curve[idx], new_curve[idx+1]
        y1_qy, y2_qy = qy_curve[idx], qy_curve[idx+1]
        
        # 线性插值求交点
        # new: y = y1_new + (y2_new - y1_new) * t
        # qy:  y = y1_qy + (y2_qy - y1_qy) * t
        # 求解: y1_new + (y2_new - y1_new) * t = y1_qy + (y2_qy - y1_qy) * t
        if (y2_new - y1_new) != (y2_qy - y1_qy):
            t = (y1_qy - y1_new) / ((y2_new - y1_new) - (y2_qy - y1_qy))
            break_even_weight = x1 + t * (x2 - x1)
            break_even_price = y1_new + t * (y2_new - y1_new)

st.subheader("价格曲线对比（同一坐标系）")

# 创建左右两列布局：左侧图表，右侧表格
col_left, col_right = st.columns([1, 1])

with col_left:
    # 创建交互式Plotly图表
    fig = go.Figure()
    
    # 添加顺丰价格曲线
    fig.add_trace(go.Scatter(
        x=x, 
        y=sf_curve,
        mode='lines+markers',
        name='顺丰价格',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6, symbol='circle'),
        hovertemplate='<b>顺丰价格</b><br>重量: %{x:.1f} kg<br>价格: %{y:.2f} 元<extra></extra>'
    ))
    
    # 添加企业价格曲线
    fig.add_trace(go.Scatter(
        x=x, 
        y=qy_curve,
        mode='lines+markers',
        name='企业价格',
        line=dict(color='#ff7f0e', width=2),
        marker=dict(size=6, symbol='square'),
        hovertemplate='<b>企业价格</b><br>重量: %{x:.1f} kg<br>价格: %{y:.2f} 元<extra></extra>'
    ))
    
    # 添加新价格曲线
    fig.add_trace(go.Scatter(
        x=x, 
        y=new_curve,
        mode='lines+markers',
        name=f'新价格(利润率{profit_pct:.1f}% + {fixed_amount:.1f}元)',
        line=dict(color='#2ca02c', width=2),
        marker=dict(size=6, symbol='triangle-up'),
        hovertemplate='<b>新价格</b><br>重量: %{x:.1f} kg<br>价格: %{y:.2f} 元<extra></extra>'
    ))
    
    # 添加价格优势转换点辅助线
    if break_even_weight is not None:
        # 添加垂直虚线（不带标注）
        fig.add_vline(
            x=break_even_weight,
            line_dash="dash",
            line_color="red",
            line_width=2
        )
        
        # 添加交点标记
        fig.add_trace(go.Scatter(
            x=[break_even_weight],
            y=[break_even_price],
            mode='markers',
            name='价格优势转换点',
            marker=dict(size=12, color='red', symbol='star', line=dict(width=2, color='darkred')),
            hovertemplate='<b>价格优势转换点</b><br>重量: %{x:.2f} kg<br>价格: %{y:.2f} 元<extra></extra>',
            showlegend=True
        ))
        
        # 添加文字标注（放在交点附近）
        fig.add_annotation(
            x=break_even_weight,
            y=break_even_price,
            text=f"价格优势转换点<br>{break_even_weight:.2f}kg<br>{break_even_price:.2f}元",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="red",
            ax=60,  # 标注框相对于点的x偏移
            ay=-100,  # 标注框相对于点的y偏移
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="red",
            borderwidth=2,
            borderpad=4,
            font=dict(size=11, color="red", family='Microsoft YaHei, SimHei, Arial')
        )
    
    # 设置图表布局
    fig.update_layout(
        title=dict(
            text=f"{div} - {dest} - {group}",
            font=dict(size=16, family='Microsoft YaHei, SimHei, Arial')
        ),
        xaxis=dict(
            title=dict(
                text='重量 (kg)',
                font=dict(size=14, family='Microsoft YaHei, SimHei, Arial')
            ),
            gridcolor='lightgray',
            gridwidth=0.5
        ),
        yaxis=dict(
            title=dict(
                text='价格 (元)',
                font=dict(size=14, family='Microsoft YaHei, SimHei, Arial')
            ),
            gridcolor='lightgray',
            gridwidth=0.5
        ),
        legend=dict(
            font=dict(size=12, family='Microsoft YaHei, SimHei, Arial'),
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        hovermode='closest',
        plot_bgcolor='white',
        height=500,
        margin=dict(l=60, r=30, t=80, b=60)
    )
    
    # 显示图表
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.write("### 数据表格")
    # 创建数据表格
    export_df = pd.DataFrame({
        "重量(kg)": x,
        "顺丰价格": sf_curve,
        "企业价格": qy_curve,
        f"新价格(利润率{profit_pct:.1f}% + {fixed_amount:.1f}元)": new_curve
    })
    # 格式化显示
    styled_df = export_df.copy()
    styled_df["顺丰价格"] = styled_df["顺丰价格"].apply(lambda v: f"{v:.2f}" if not np.isnan(v) else "—")
    styled_df["企业价格"] = styled_df["企业价格"].apply(lambda v: f"{v:.2f}" if not np.isnan(v) else "—")
    styled_df[f"新价格(利润率{profit_pct:.1f}% + {fixed_amount:.1f}元)"] = styled_df[f"新价格(利润率{profit_pct:.1f}% + {fixed_amount:.1f}元)"].apply(lambda v: f"{v:.2f}" if not np.isnan(v) else "—")
    
    st.dataframe(styled_df, use_container_width=True, height=400)

# 下载按钮
st.download_button("下载当前曲线CSV", data=export_df.to_csv(index=False).encode("utf-8-sig"),
                   file_name=f"曲线_{div}_{group}.csv", mime="text/csv")

st.caption("CSV需包含：分区/目的地/价格对比 + 6个重量点(1/2/3/4/5/15kg)；程序基于这些点位在[1,15]kg做分段线性插值。")

# ==================== 价格优势转换点汇总表格 ====================
st.markdown("---")
st.subheader("价格优势转换点汇总分析")

def calculate_break_even_point(sf_data, qy_data, profit_rate, fixed_amt, weight_range):
    """
    计算价格优势转换点
    返回: (转换点重量, 状态) 其中状态可能是具体重量值、'全优'或'全劣'
    """
    if sf_data.empty or qy_data.empty:
        return None, "数据缺失"
    
    # 构建曲线
    x_grid = np.arange(weight_range[0], weight_range[1] + 0.0001, 0.5)
    
    def build_curve_local(df_points):
        pts = {int(r["重量kg"]): float(r["价格"]) 
               for _, r in df_points.iterrows() 
               if float(r["重量kg"]) in (1, 2, 3, 4, 5, 15)}
        if not pts:
            return None
        xs = sorted(pts.keys())
        ys = [pts[x] for x in xs]
        if any(pd.isna(v) for v in ys):
            return None
        return np.interp(x_grid, xs, ys)
    
    sf_curve_local = build_curve_local(sf_data)
    qy_curve_local = build_curve_local(qy_data)
    
    if sf_curve_local is None or qy_curve_local is None:
        return None, "数据缺失"
    
    # 修改：新价格计算公式包含利润率和固定金额
    new_curve_local = sf_curve_local * (1.0 + profit_rate / 100.0) + fixed_amt
    
    # 计算差值
    diff = new_curve_local - qy_curve_local
    
    # 判断全面优势或全面劣势
    if np.all(diff <= 0):
        return None, "全部占优"
    if np.all(diff >= 0):
        return None, "全部劣势"
    
    # 查找交点
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_changes) == 0:
        return None, "无交点"
    
    # 线性插值计算第一个交点
    idx = sign_changes[0]
    x1, x2 = x_grid[idx], x_grid[idx + 1]
    y1_new, y2_new = new_curve_local[idx], new_curve_local[idx + 1]
    y1_qy, y2_qy = qy_curve_local[idx], qy_curve_local[idx + 1]
    
    if (y2_new - y1_new) != (y2_qy - y1_qy):
        t = (y1_qy - y1_new) / ((y2_new - y1_new) - (y2_qy - y1_qy))
        break_even_weight = x1 + t * (x2 - x1)
        return break_even_weight, "转换点"
    
    return None, "无交点"

# 生成汇总表格
st.write("#### 各区域价格优势转换点汇总")
st.write(f"当前利润率：{profit_pct:.1f}% | 固定金额：{fixed_amount:.1f}元 | 分析重量范围：2-15kg")

# 获取所有分区和目的地
all_divisions = sort_divisions(long_df["分区"].dropna().unique().tolist())
all_groups = ["日均30-50单", "日均50-100单", "日均100-200单", "日均200-300单", "日均300-500单", "日均500单以上"]

# 构建汇总数据
summary_data = []
for div_item in all_divisions:
    dests_list = long_df[long_df["分区"] == div_item]["目的地"].dropna().astype(str).unique().tolist()
    for dest_item in sorted(dests_list):
        row_data = {"分区": div_item, "目的地": dest_item}
        for group_item in all_groups:
            sub_data = long_df[
                (long_df["分区"] == div_item) & 
                (long_df["目的地"].astype(str) == str(dest_item)) & 
                (long_df["日均区间"] == group_item)
            ]
            sf_data = sub_data[sub_data["价格对比"] == "顺丰价格"].dropna(subset=["重量kg", "价格"])
            qy_data = sub_data[sub_data["价格对比"] == "企业价格"].dropna(subset=["重量kg", "价格"])
            
            # 修改：传入fixed_amount参数
            weight, status = calculate_break_even_point(sf_data, qy_data, profit_pct, fixed_amount, (2.0, 15.0))
            
            if status == "转换点" and weight is not None:
                row_data[group_item] = f"{weight:.2f}kg"
            else:
                row_data[group_item] = status
        
        summary_data.append(row_data)

# 创建DataFrame
summary_df = pd.DataFrame(summary_data)

# 显示表格
st.dataframe(summary_df, use_container_width=True, height=400)

# 添加说明
st.caption("""
**说明：**
- **具体重量值（如 5.23kg）**：表示在该重量点新价格与企业价格持平，超过该重量具有价格优势，低于该重量处于价格劣势
- **全优**：在整个重量范围内（2-15kg），新价格始终低于企业价格，全程具有价格优势
- **全劣**：在整个重量范围内（2-15kg），新价格始终高于企业价格，全程处于价格劣势
- **数据缺失**：该区域该日均区间没有完整的价格数据
""")

# 下载汇总表格按钮
st.download_button(
    "下载价格优势转换点汇总表",
    data=summary_df.to_csv(index=False).encode("utf-8-sig"),
    file_name=f"价格优势转换点汇总_利润率{profit_pct:.0f}%_固定金额{fixed_amount:.1f}元.csv",
    mime="text/csv"
)
