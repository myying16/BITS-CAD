import numpy as np
#EOS: 序列结束标记
#SOL: 草图开始标记
#Ext: 拉伸操作
#C: 贝塞尔曲线
CAD_COMMANDS = ['Line', 'Arc', 'Circle', 'EOS', 'SOL', 'Ext']
CAD_LINE_IDX = CAD_COMMANDS.index('Line')   #2line end-point
CAD_ARC_IDX = CAD_COMMANDS.index('Arc') #4=arc end-point+sweep angle+counter-clockwise flag
CAD_CIRCLE_IDX = CAD_COMMANDS.index('Circle') #3
CAD_EOS_IDX = CAD_COMMANDS.index('EOS')
CAD_SOL_IDX = CAD_COMMANDS.index('SOL')
CAD_EXT_IDX = CAD_COMMANDS.index('Ext')
#创建新体；合并操作-并集；切割操作-差集；相交参数-交集
CAD_EXTRUDE_OPERATIONS = ["NewBodyFeatureOperation", "JoinFeatureOperation",
                      "CutFeatureOperation", "IntersectFeatureOperation"]
#单侧拉伸；对称拉伸；双侧拉伸
CAD_EXTENT_TYPE = ["OneSideFeatureExtentType", "SymmetricFeatureExtentType",
               "TwoSidesFeatureExtentType"]
PAD_VAL = -1 #填充值，用于序列填充
#定义CAD命令的参数数量
CAD_N_ARGS_SKETCH = 5 # 草图参数sketch parameters: x, y, alpha角度参数, f标志位, r半径
CAD_N_ARGS_PLANE = 3 # 平面方向参数sketch plane orientation: θ, φ, γ
CAD_N_ARGS_TRANS = 4 # 变换参数：原点坐标+边界框大小sketch plane origin + sketch bbox size: p_x, p_y, p_z, s
CAD_N_ARGS_EXT_PARAM = 4 # 拉伸参数extrusion parameters: e1,e2拉伸距离, b布尔操作类型索引, u拉伸方式索引
CAD_N_ARGS_EXT = CAD_N_ARGS_PLANE + CAD_N_ARGS_TRANS + CAD_N_ARGS_EXT_PARAM #3+4+4=11
CAD_N_ARGS = CAD_N_ARGS_SKETCH + CAD_N_ARGS_EXT #16
#[x, y, α, f, r,    θ, φ, γ,    px, py, pz, s,   e1, e2, b, u]

SVG_COMMANDS = ['SOS', 'EOS', 'L', 'C']
SVG_SOS_IDX = SVG_COMMANDS.index('SOS')
SVG_EOS_IDX = SVG_COMMANDS.index('EOS')
SVG_L_IDX = SVG_COMMANDS.index('L')
SVG_C_IDX = SVG_COMMANDS.index('C')
#定义SVG命令参数数量
SVG_N_ARGS_LINE = 4 # start=(x1, y1), end=(x2, y2)
SVG_N_ARGS_BEZIERCURVE = 8 # 贝塞尔曲线start=(x1, y1), control1=(cx1,cy1), control2=(cx2,cy2), end=(x2, y2)
SVG_N_ARGS = 8 # 统一参数维度shared parameters

SVG_SOS_VEC = np.array([SVG_SOS_IDX, *([PAD_VAL] * SVG_N_ARGS)])
SVG_EOS_VEC = np.array([SVG_EOS_IDX, *([PAD_VAL] * SVG_N_ARGS)])


CAD_SOL_VEC = np.array([CAD_SOL_IDX, *([PAD_VAL] * CAD_N_ARGS)])
CAD_EOS_VEC = np.array([CAD_EOS_IDX, *([PAD_VAL] * CAD_N_ARGS)])
CAD_CMD_ARGS_MASK = np.array([[1, 1, 0, 0, 0, *[0]*CAD_N_ARGS_EXT],  # line
                          [1, 1, 1, 1, 0, *[0]*CAD_N_ARGS_EXT],  # arc
                          [1, 1, 0, 0, 1, *[0]*CAD_N_ARGS_EXT],  # circle
                          [0, 0, 0, 0, 0, *[0]*CAD_N_ARGS_EXT],  # EOS
                          [0, 0, 0, 0, 0, *[0]*CAD_N_ARGS_EXT],  # SOL
                          [*[0]*CAD_N_ARGS_SKETCH, *[1]*CAD_N_ARGS_EXT]]) # Extrude拉伸命令使用所有拉伸参数

SVG_CMD_ARGS_MASK = np.array([[*[0]*SVG_N_ARGS], # SOS无有效参数
                             [*[0]*SVG_N_ARGS], # EOS无有效参数
                             [1, 1, 0, 0, 0, 0, 1, 1], # L
                             [*[1]*SVG_N_ARGS]]) # C贝塞尔曲线使用所有参数

CAD_NORM_FACTOR = 0.75 # scale factor for normalization to prevent overflow during augmentation

CAD_MAX_N_EXT = 10 # maximum number of extrusion最大拉伸操作数
CAD_MAX_N_LOOPS = 6 # maximum number of loops per sketch每个草图最大环数
CAD_MAX_N_CURVES = 15 # maximum number of curves per loop每个环最大曲线数

CAD_MAX_TOTAL_LEN = 60 # maximum cad sequence length
SVG_MAX_TOTAL_LEN = 100 # maximum svg sequence length
ARGS_DIM = 256