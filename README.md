# 语音识别
数字信号处理课程项目

项目时间：2018年6月

本项目使用语音生成语谱图后，对语谱图使用ResNet进行训练分类，达成区分20个中英文词汇的语音的效果
这些词汇分别为
['语音','余音','识别','失败','中国','忠告','北京','背景','上海','商行','复旦','饭店',
     'speech','speaker','signal','file','print','open','close','project']

# 目录结构
report.pdf 电子版报告
resnet.h5 训练好的resnet模型
src 全部代码
	-draw.py - 绘制模型结构图
	-fft.py - 实现快速傅里叶变换的函数
	-get_data.py - 处理、读取语谱图并将之划分为训练和测试集的程序
	-main.py - 图形化的预测程序
	-predict.py - 命令行版的预测程序
	-record.py - 录音程序
	-resnet.py 构建深度残差网络的程序
	-spec.py 实现绘制灰度语谱图的程序
	-specgram.py 将采集的所有数据绘制成语谱图，其中有注释直接用matplotlib绘制的部分
	-train_test.py 训练并测试网络的程序