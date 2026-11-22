# video-color-band-mitigate

## 题面

暗部渐变一压就出色带，加抖动后颗粒又会闪、导出还变形：tpdf_noise 用均匀分布冒充三角分布、颗粒感弱；tpdf_noise 的幅度开到 2 LSB、暗部噪声过强；blue_noise_1d 用白噪声冒充蓝噪声、相邻相关性为 0；apply_dither 先量化再加噪声、抖动无效；apply_dither 的噪声每帧复用同一 seed、静止时噪点不爬行；quantize 用 floor 代替最近舍入、整体偏暗半 LSB；deband_gradient 把步数当位深的指数、色带模拟过度；band_variance 用整图方差、被边缘带偏；temporal_stability 用同一帧比较、恒为 1；validate 不检查 bits<=0、除零；deband/dither.py 是核心，tests/test_dither.py 15个红。要修的是：把均匀噪声换成 1 LSB 的 TPDF、让蓝噪声相邻负相关、先加噪再量化并按帧派生 seed、把 floor 改成最近舍入、按电平个数而不是 2 的幂模拟色带、用带内中值高通替代整图方差、先扣掉逐像素时间均值再算帧间相关性、补上 bits<=0 的校验，直到 `python -m pytest tests -q` 输出 15 passed。

## 运行方式

```powershell
cd D:\gsb-20261119\video-color-band-mitigate
python -m pytest tests -q
```

初始失败环境应恰好输出 `15 failed`，没有 collection/import/syntax error；15 条失败全部来自 `deband/dither.py` 中 8 个桩的 `NotImplementedError`。修复实现后应全部转绿。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

`tests/test_dither.py` 是评测基准，**不可修改**；只允许修改 `deband/dither.py`。实现只使用 Python 标准库（`math`）与 pytest，离线运行，不使用随机数；噪声必须由显式整数 `seed` 经确定性哈希生成。图像为扁平的线性光 `list[float]`，样本范围 `0..1`；位深 `bits` 为 `1..16`；浮点比较容差为 `1e-9`。
