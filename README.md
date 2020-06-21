注意！因上传大小限制，该Git仓库没有上传模型文件到x64文件夹中

C++ 调用 Python 教程：https://blog.csdn.net/xiaomu_347/article/details/81040855 



Python脚本注意事项：

模型文件，数据集，需要为绝对路径

Python脚本返回值必须为元组，使用temp作为临时拼凑变量



from_tensor_slices（参数：图片路径集合）

from_tensors（参数：单张图片路径）



model.evaluate(数据集，测试集)  返回准确值和损失率

model.predict(数据集) 返回预测结果（预测结果需要变为整型 np.argmax(results, axis=-1)）



