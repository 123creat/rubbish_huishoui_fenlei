# rubbish_huishoui_fenlei
#这是一个python的垃圾分类邮箱回收的教程
#接下来会教你怎么做


首先你需要下载jupyter-notebook和acanda环境
这里是acanda下载路径：https://blog.csdn.net/m0_61607990/article/details/129531686?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_utm_term~default-1-129531686-blog-125318924.235^v43^pc_blog_bottom_relevance_base2&spm=1001.2101.3001.4242.2&utm_relevant_index=4
这里是jupyter-notebook环境：https://blog.csdn.net/m0_68678046/article/details/129703799?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171348732416800188553500%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=171348732416800188553500&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-129703799-null-null.142^v100^pc_search_result_base2&utm_term=jupyter%20notebook&spm=1018.2226.3001.4187



下载完环境之后你需要：安装一下对应的包
这里是下载的镜像源路径：pip install  -i https://mirrors.aliyun.com/pypi/simple/ scikit-learn jieba
pip install numpy

接下来是步骤
首先打开jupyter-notebook
![image](https://github.com/123creat/rubbish_huishoui_fenlei/assets/116633051/088b7be2-8c6a-4cd3-a835-230a25cddadc)

接下来是分词器：
###################################################################################################
#计外一班-熊卫涛
# 没装环境的需要先装一下环境，pip install  -i https://mirrors.aliyun.com/pypi/simple/ scikit-learn jieba
import re
import jieba
import codecs
import os
# 去掉非中文字符
def clean_str(string):
    string = re.sub(r"[^\u4e00-\u9fff]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()



def get_data_in_a_file(original_path, save_path='all_emails.txt'):
    files = os.listdir(original_path)
    for file in files:
        if os.path.isdir(original_path + '/' + file):
                get_data_in_a_file(original_path + '/' + file, save_path=save_path)
        else:
            email = ''
            # 注意要用 'ignore'，不然会报错
            f = codecs.open(original_path + '/' + file, 'r', 'gbk', errors='ignore')
            # lines = f.readlines()
            for line in f:
                line = clean_str(line)
                email += line
            f.close()
           
            f = open(save_path, 'a', encoding='utf8')
            email = [word for word in jieba.cut(email) if word.strip() != '']
            f.write(' '.join(email) + '\n')

def get_label_in_a_file(original_path, save_path='all_emails.txt',flag='0'):
    # 统计文件个数
    count = 0
    for _, _, files in os.walk(original_path):
        count += len(files)
    # 为每个文件写入标签
    label_list = [flag] * count
    f = open(save_path, 'a', encoding='utf8')
    f.write('\n'.join(label_list))
    f.close()



if __name__ == '__main__':
    # step 1
    print('Storing emails in a file ...')
    #这里无论是使用绝对路径还是相对路径都是生成在当前项目文件夹的同一级，不清楚为什么
    #但是建议你修改一下
    #我猜测这里可能是将左边路径里的文件，写入到右边
    get_data_in_a_file('C:/Users/86131/Desktop/VS/PY/py-project-lajihoushou/data/normal', save_path='all_emails.txt')
    get_data_in_a_file('C:/Users/86131/Desktop/VS/PY/py-project-lajihoushou/data/spam', save_path='all_emails.txt')
    print('Store emails finished !')
 
    # step 2
    print('Storing labels in a file ...')
    #这里无论是使用绝对路径还是相对路径都是生成在当前项目文件夹的同一级，不清楚为什么
    #但是建议你修改一下
    #我猜测这里可能是将左边路径里的文件，写入到右边
    get_label_in_a_file('C:/Users/86131/Desktop/VS/PY/py-project-lajihoushou/data/normal', save_path='label.txt',flag='1')
    get_label_in_a_file('C:/Users/86131/Desktop/VS/PY/py-project-lajihoushou/data/spam', save_path='label.txt',flag='0')
    print('Store labels finished !')



####################################################################
########################################################################
#####################################################################



运行完分词器你的项目文件夹里会多出两个文件（注意，代码中的路径需要修改成你自己的）


接下来是测试文件
#######################################################
#计外一班-熊卫涛
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm, ensemble, naive_bayes
from sklearn import metrics
import numpy as np
#没有numpy的需要先装一下

def get_data_tf_idf(email_file_name):
    vectoring = TfidfVectorizer(input='content', tokenizer=tokenizer_space, analyzer='word')
    content = open(email_file_name, 'r', encoding='utf8').readlines()
    x = vectoring.fit_transform(content)
    return x, vectoring

def get_label_list(label_file_name):
    with open(label_file_name, 'r', encoding='utf-8') as f:
        label_list = [line.strip() for line in f]  
    return label_list

def tokenizer_space(line):
    return [li for li in line.split() if li.strip() != '']

#这里用的是自己所生成的文件的绝对路径，这样文件的位置固定
np.random.seed(1)
email_file_name = 'C:/Users/86131/Desktop/VS/PY/all_emails.txt'
label_file_name = 'C:/Users/86131/Desktop/VS/PY/label.txt'

x, vectoring = get_data_tf_idf(email_file_name)
y = get_label_list(label_file_name)
index = np.arange(len(y))  
y = np.array(y)
np.random.shuffle(index)
y = y[index]
x = x[index]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)
clf = svm.LinearSVC()
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
print('classification_report\n', metrics.classification_report(y_test, y_pred, digits=4))
print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
################################################################


运行完test.py文件后，
当你得到类似于这样的结果，就可以保存你的文件上交了：

classification_report
               precision    recall  f1-score   support

           0     0.9906    0.9956    0.9931      1588
           1     0.9949    0.9891    0.9920      1380

    accuracy                         0.9926      2968
   macro avg     0.9927    0.9924    0.9925      2968
weighted avg     0.9926    0.9926    0.9926      2968

Accuracy: 0.9925876010781671

------------------------------------------------------------
