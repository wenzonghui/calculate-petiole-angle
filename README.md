# calculate-petiole-angle
本项目用于对大豆或其他植物生长过程中的叶柄夹角进行计算。该方法包括【半自动方法】和【全自动方法】。

## 声明
如果您感觉我们的方法对您有所帮助，请引用我们的文章。

## TODO List
- 2025/5/27 项目创建
- 2025/6/01  发布半自动代码
- 2025/6/15 发布全自动代码
- 2025/6/30 发布封装后的程序应用

## 半自动方法
### 关键点标注
使用者须运行程序前对处理涉及的图片进行关键点标注：第一张图需要标注叶柄夹角的三个关键点，后续图片只需要标注叶柄的关键点即可，其余两个关键点默认使用第一张的位置。

标注颜色使用：红色#FF0000（255，0， 0）

例如：

第一张标注样例

后续图片标注样例

### 植株剪切
此步骤针对于同一张图片中有多个计算植株的情况。
1. 需要处理的图片全部复制进项目文件夹的 images 下
2. 在运行主代码中修改需要分拆的植株横坐标像素
3. 运行程序会将多个植株剪切进多个文件夹中，后续该植株的叶柄夹角会保存在对应植株的文件夹中

### 叶柄角度计算
1. 运行程序，程序会依次计算本项目下多个植株的叶柄夹角，并将数据保存至对应的文件夹下

## 全自动方法
1. 使用者需要将待处理的图片复制进项目下的 images 文件夹内
2. 使用者需要修改分拆的植株横坐标像素
3. 运行程序即可，叶柄夹角数据将保存至对应植株的文件夹下
