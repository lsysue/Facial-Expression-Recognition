# coding=utf-8
"""

本文件用于自动检验对测试集的分类结果的文件格式
分类结果的文件名需为 prediction.csv，即下方 filename 常量

"""
import os

filename = 'prediction.csv'
emotions_all = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
test_total_num = 10761


def check_prediction():
    assert os.path.exists(filename), "请将分类结果 prediction.csv 放于此代码相同目录，重新测试。"

    preds = open(filename, 'r').readlines()
    sample_count = 0
    for i, result in enumerate(preds):
        result = result.strip()
        if len(result) == 0:
            print('Warning: 第 %d 行是空行' % (i + 1))
            continue
        try:
            sample_name, emotion = result.split(',')

            success = True
            """ 检查样本名 """
            if not sample_name.endswith('.jpg'):
                print('Error: 第 %d 行样本名后缀不对，应该是小写.jpg' % (i + 1))
                success = False
            elif not sample_name.startswith('test_'):
                print('Error: 第 %d 行样本名应该以 test_ 开头' % (i + 1))
                success = False
            num = int(sample_name[5:10])
            if num < 0 or num >= test_total_num:
                print('Error: 第 %d 行样本不属于 test_00000.jpg 至 test_10760.jpg 中' % (i + 1))
                success = False

            """ 检查预测值 """
            if emotion not in emotions_all:
                print('Error: 第 %d 行预测值 %s 不属于表情标签之一，表情标签有：%s' % (i + 1, emotion, str(emotions_all)))
                success = False

            """ 登记检测结果 """
            if success:
                sample_count += 1
        except ValueError:
            if '，' in result:
                print('Error: 第 %d 行错误使用了中文逗号，应用英文逗号' % (i + 1))
            elif len(result.split(',')) < 2:
                print('Error: 第 %d 行缺少逗号' % (i + 1))
            elif len(result.split(',')) > 2:
                print('Error: 第 %d 行逗号过多' % (i + 1))
            else:
                print('Error: 第 %d 行有字符错误，可能是数字部分位数不对' % (i + 1))

    if sample_count == test_total_num:
        print("检测通过")
    else:
        print("检测不通过，合法结果只有%d条，不足%d条。" % (sample_count, test_total_num))


if __name__ == "__main__":
    check_prediction()
