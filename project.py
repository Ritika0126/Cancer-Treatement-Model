import pandas as pd
import matplotlib.pyplot as plt
import re
import time
import warnings
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import cross_val_score
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
from scipy.sparse import hstack
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
#from sklearn.cross_validation import StratifiedKFold
from collections import Counter, defaultdict
from sklearn.calibration import CalibratedClassifierCV
import math
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import normalized_mutual_info_score
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")
from mlxtend.classifier import StackingClassifier
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import cross
warnings.filterwarnings("ignore")
#print("hello")
data_variants=pd.read_csv('training_variants')
data_text=pd.read_csv("training_text",sep="\|\|",engine="python",names=["ID","TEXT"],skiprows=1)
#print(data_variants.head(3))
#print(data_text.head(3))
stop_words=set(stopwords.words('english'))
#data preprocessing
def data_text_preprocess(total_text,ind,col):
    if type(total_text) is not int:
        string=""
        total_text=re.sub('[^a-zA-Z0-9\n)]',' ',str(total_text))        #replace special characters with space
        total_text=re.sub('\s+',' ',str(total_text))       #replace multi spaces with singl
        total_text=total_text.lower()
        for word in total_text.split():
            if not word in stop_words:
                string += word + " "
        data_text[col][ind]=string
for index,row in data_text.iterrows():
    if type(row['TEXT']) is str:
        data_text_preprocess(row['TEXT'],index,'TEXT')
print(data_text.head(3))
result=pd.merge(data_variants,data_text,on='ID',how='left')
print(result.shape)
result[result.isnull().any(axis=1)]
result.loc[result['TEXT'].isnull(),'TEXT']=result['Gene']+ ' '+ result['Variation']
result[result.isnull().any(axis=1)]
y_true=result['Class'].values
result.Gene=result.Gene.str.replace('\s+','_')
result.Variation=result.Variation.str.replace('\s+','_')
X_train,test_df,y_train,y_test=train_test_split(result,y_true,stratify=y_true,test_size=0.2)
train_df,cv_df,y_train,y_cv=train_test_split(X_train,y_train,stratify=y_train,test_size=0.2)
"""
train_class_distribution=train_df['Class'].value_counts().sortlevel()
test_class_distribution=test_df['Class'].value_counts().sortlevel()
cv_class_distribution=cv_df['Class'].value_counts().sortlevel()
#did this to check distribution od data for ezch class

print(train_class_distribution)
my_colors='rgbkymc'
train_class_distribution.plot(kind='bar')
plt.xlabel('Class')
plt.ylabel('Number of data points')
plt.title('Distribution')
plt.grid()
plt.show()
"""
#check distribution in all sets i.e train ,test,cv and plot graph and find percentage
test_data_len=test_df.shape[0]
cv_data_len=cv_df.shape[0]
#create random model for apllying log loss

cv_predicted_y=np.zeros((cv_data_len,9))
for i in range(cv_data_len):
    rand_probs=np.random.rand(1,9)
    cv_predicted_y[i]=((rand_probs/sum(sum(rand_probs)))[0])
print("log loss on cv data using random model",log_loss(y_cv,cv_predicted_y,eps=1e-15))
     
test_predicted_y=np.zeros((test_data_len,9))
for i in range(test_data_len):
    rand_probs=np.random.rand(1,9)
    test_predicted_y[i]=((rand_probs/sum(sum(rand_probs)))[0])
print("log loss on test data using random model",log_loss(y_test,test_predicted_y,eps=1e-15))

predicted_y=np.argmax(test_predicted_y,axis=1)
print(predicted_y)

predicted_y=predicted_y+1

C=confusion_matrix(y_test,predicted_y)
labels=[1,2,3,4,5,6,7,8,9]
plt.figure(figsize=(20,7))
sns.heatmap(C,annot=True,cmap='YlGnBu',fmt=".3f",xticklabels=labels,yticklabels=labels)
plt.xlabel('Predicted class')
plt.ylabel('Original class')
plt.show()

#precision mattrix....colmn
B=(C/C.sum(axis=0))
plt.figure(figsize=(20,7))
sns.heatmap(B,annot=True,cmap='YlGnBu',fmt=".3f",xticklabels=labels,yticklabels=labels)
plt.xlabel('Predicted class')
plt.ylabel('Original class')
plt.show()

#recall matrix...row

A=(((C.T/C.sum(axis=1))).T)
plt.figure(figsize=(20,7))
sns.heatmap(A,annot=True,cmap='YlGnBu',fmt=".3f",xticklabels=labels,yticklabels=labels)
plt.xlabel('Predicted class')
plt.ylabel('Original class')
plt.show()

#eVALUATING EACH COLUMN

unique_genes=train_df['Gene'].value_counts()
print('Number of unique Genes:',unique_genes.shape[0])
print(unique_genes.head(10))

s=sum(unique_genes.values)
h=unique_genes.values/s
c=np.cumsum(h)
plt.plot(c,label='Cummulative distribution')
plt.grid()
plt.legend()
plt.show()

gene_vectorizer=CountVectorizer()
train_gene_feature_onehotCoding=gene_vectorizer.fit_transform(train_df['Gene'])
test_gene_feature_onehotCoding=gene_vectorizer.transform(test_df['Gene'])
cv_gene_feature_onehotCoding=gene_vectorizer.transform(cv_df['Gene'])

print(train_gene_feature_onehotCoding.shape)

#response encoding with laplace smoothing
def get_gv_fea_dict(alpha,feature,df):
     value_count=train_df[feature].value_counts()
     gv_dict=dict()
     for i,denominator in value_count.items():
         vec=[]
         for k in range(1,10):
             cls_cnt=train_df.loc[(train_df['Class']==k)&(train_df[feature]==i)]
             vec.append((cls_cnt.shape[0] + alpha*10)/(denominator +90*alpha))
         gv_dict[i]=vec
     return gv_dict
             
    


def get_gv_feature(alpha,feature,df):
    gv_dict=get_gv_fea_dict(alpha,feature,df)
    value_count=train_df[feature].value_counts()
    gv_fea=[]
    for index,row in df.iterrows():
        if row[feature] in dict(value_count).keys():
            gv_fea.append(gv_dict[row[feature]])
        else:
            gv_fea.append([1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9])
    return gv_fea

alpha=1

train_gene_feature_responseCoding=np.array(get_gv_feature(alpha,"Gene",train_df))
test_gene_feature_responseCoding=np.array(get_gv_feature(alpha,"Gene",test_df))
cv_gene_feature_responseCoding=np.array(get_gv_feature(alpha,"Gene",cv_df))

print(train_gene_feature_responseCoding.shape)

alpha=[10**x for x in range(-5,1)]

#use sbd and callibrated classifier
cv_log_error_array=[]

for i in alpha:
    clf=SGDClassifier(alpha=i,penalty='l2' ,loss='log',random_state=42)
    clf.fit(train_gene_feature_onehotCoding,y_train)
    sig_clf=CalibratedClassifierCV(clf,method='sigmoid')
    sig_clf.fit(train_gene_feature_onehotCoding,y_train)
    predict_y=sig_clf.predict_proba(cv_gene_feature_onehotCoding)
    cv_log_error_array.append(log_loss(y_cv, predict_y,labels=clf.classes_,eps=1e-15))
    print('For alpha= ',i,' log loss is ',log_loss(y_cv,predict_y,labels=clf.classes_,eps=1e-15))

fig,ax=plt.subplots()
ax.plot(alpha,cv_log_error_array,c='g')
for i, txt in enumerate(np.round_(cv_log_error_array,3)):
    ax.annotate((alpha[i],np.round(txt,3)),(alpha[i],cv_log_error_array[i]))
plt.grid()
plt.title("cross validation")
plt.xlabel("alpha i")
plt.ylabel("error ")
plt.show()

best_alpha=np.argmin(cv_log_error_array)
clf=SGDClassifier(alpha=alpha[best_alpha],penalty='l2',loss='log',random_state=42)
clf.fit(train_gene_feature_onehotCoding,y_train)
sig_clf=CalibratedClassifierCV(clf,method='sigmoid')
sig_clf.fit(train_gene_feature_onehotCoding,y_train)

predict_y=sig_clf.predict_proba(train_gene_feature_onehotCoding)
print('For values of best alpha ',alpha[best_alpha],' train log loss is ',log_loss(y_train,predict_y,labels=clf.classes_,eps=1e-15))

predict_y=sig_clf.predict_proba(cv_gene_feature_onehotCoding)
print('For values of best alpha ',alpha[best_alpha],' cv log loss is ',log_loss(y_cv,predict_y,labels=clf.classes_,eps=1e-15))

predict_y=sig_clf.predict_proba(test_gene_feature_onehotCoding)
print('For values of best alpha ',alpha[best_alpha],' test log loss is ',log_loss(y_test,predict_y,labels=clf.classes_,eps=1e-15))

test_coverage=test_df[test_df['Gene'].isin(list(set(train_df['Gene'])))].shape[0]
cv_coverage=cv_df[cv_df['Gene'].isin(list(set(train_df['Gene'])))].shape[0]

print('1.in test data ',test_coverage,'out of ',test_df.shape[0],':',(test_coverage/test_df.shape[0])*100)      #check overlaps 

print('2.in cv data ',cv_coverage,'out of ',cv_df.shape[0],':',(cv_coverage/cv_df.shape[0])*100)

#evaluate variation column
unique_variations=train_df['Variation'].value_counts()
print('Number of unique variations:',unique_variations.shape[0])
print(unique_variations.head(10))

s=sum(unique_variations.values)
h=unique_variations.values/s
c=np.cumsum(h)
plt.plot(c,label='Cummulative distribution')
plt.grid()
plt.legend()
plt.show()
    
variation_vectorizer=CountVectorizer()
train_variation_feature_onehotCoding=variation_vectorizer.fit_transform(train_df['Variation'])
test_variation_feature_onehotCoding=variation_vectorizer.transform(test_df['Variation'])
cv_variation_feature_onehotCoding=variation_vectorizer.transform(cv_df['Variation'])

print(train_variation_feature_onehotCoding.shape)                          


alpha=1

train_variation_feature_responseCoding=np.array(get_gv_feature(alpha,"Variation",train_df))
test_variation_feature_responseCoding=np.array(get_gv_feature(alpha,"Variation",test_df))
cv_variation_feature_responseCoding=np.array(get_gv_feature(alpha,"Variation",cv_df))

print(train_variation_feature_responseCoding.shape)

alpha=[10**x for x in range(-5,1)]

#use sbd and callibrated classifier
cv_log_error_array=[]

for i in alpha:
    clf=SGDClassifier(alpha=i,penalty='l2' ,loss='log',random_state=42)
    clf.fit(train_variation_feature_onehotCoding,y_train)
    sig_clf=CalibratedClassifierCV(clf,method='sigmoid')
    sig_clf.fit(train_variation_feature_onehotCoding,y_train)
    predict_y=sig_clf.predict_proba(cv_variation_feature_onehotCoding)
    cv_log_error_array.append(log_loss(y_cv, predict_y,labels=clf.classes_,eps=1e-15))
    print('For alpha= ',i,' log loss is ',log_loss(y_cv,predict_y,labels=clf.classes_,eps=1e-15))

fig,ax=plt.subplots()
ax.plot(alpha,cv_log_error_array,c='g')
for i, txt in enumerate(np.round_(cv_log_error_array,3)):
    ax.annotate((alpha[i],np.round(txt,3)),(alpha[i],cv_log_error_array[i]))
plt.grid()
plt.title("cross validation")
plt.xlabel("alpha i")
plt.ylabel("error ")
plt.show()

best_alpha=np.argmin(cv_log_error_array)
clf=SGDClassifier(alpha=alpha[best_alpha],penalty='l2',loss='log',random_state=42)
clf.fit(train_variation_feature_onehotCoding,y_train)
sig_clf=CalibratedClassifierCV(clf,method='sigmoid')
sig_clf.fit(train_variation_feature_onehotCoding,y_train)

predict_y=sig_clf.predict_proba(train_variation_feature_onehotCoding)
print('For values of best alpha ',alpha[best_alpha],' train log loss is ',log_loss(y_train,predict_y,labels=clf.classes_,eps=1e-15))

predict_y=sig_clf.predict_proba(cv_variation_feature_onehotCoding)
print('For values of best alpha ',alpha[best_alpha],' cv log loss is ',log_loss(y_cv,predict_y,labels=clf.classes_,eps=1e-15))

predict_y=sig_clf.predict_proba(test_variation_feature_onehotCoding)
print('For values of best alpha ',alpha[best_alpha],' test log loss is ',log_loss(y_test,predict_y,labels=clf.classes_,eps=1e-15))

test_coverage=test_df[test_df['Variation'].isin(list(set(train_df['Variation'])))].shape[0]
cv_coverage=cv_df[cv_df['Variation'].isin(list(set(train_df['Variation'])))].shape[0]

print('1.in test data ',test_coverage,'out of ',test_df.shape[0],':',(test_coverage/test_df.shape[0])*100)      #check overlaps 

print('2.in cv data ',cv_coverage,'out of ',cv_df.shape[0],':',(cv_coverage/cv_df.shape[0])*100)

def extract_dictionary_paddle(cls_text):
    dictionary=defaultdict(int)
    for index,row in cls_text.iterrows():
        for word in row['TEXT'].split():
            dictionary[word]+=1
    return dictionary

import math

def get_text_responseCoding(df):
    text_feature_responseCoding=np.zeros((df.shape[0],9))
    for i in range(0,9):
        row_index=0
        for index,row in df.iterrows():
            sum_prob=0
            for word in row['TEXT'].split():
                sum_prob+=math.log(((dict_list[i].get(word,0)+10)/(total_dict.get(word,0)+90)))
            text_feature_responseCoding[row_index][i]=math.exp(sum_prob/len(row['TEXT'].split()))
            row_index+=1
    return text_feature_responseCoding


text_vectorizer=CountVectorizer(min_df=3)
train_text_feature_onehotCoding=text_vectorizer.fit_transform(train_df['TEXT'])

train_text_features=text_vectorizer.get_feature_names()
train_text_fea_counts=train_text_feature_onehotCoding.sum(axis=0).A1

text_fea_dict=dict(zip(list(train_text_features),train_text_fea_counts))

print("Total number of unique words in train data: ",len(train_text_features))

dict_list=[]
for i in range(1,10):
    cls_text=train_df[train_df['Class']==i]
    dict_list.append(extract_dictionary_paddle(cls_text))
total_dict=extract_dictionary_paddle(train_df)

confuse_array=[]
for i in train_text_features:
    ratios=[]
    max_val=-1
    for j in range(0,9):
        ratios.append((dict_list[j][i]+10)/(total_dict[i]+90))
    confuse_array.append(ratios)
confuse_array=np.array(confuse_array)

train_text_feature_responseCoding=get_text_responseCoding(train_df)
test_text_feature_responseCoding=get_text_responseCoding(test_df)
cv_text_feature_responseCoding=get_text_responseCoding(cv_df)

train_text_feature_responseCoding=(train_text_feature_responseCoding.T/train_text_feature_responseCoding.sum(axis=1)).T
test_text_feature_responseCoding=(test_text_feature_responseCoding.T/test_text_feature_responseCoding.sum(axis=1)).T
cv_text_feature_responseCoding=(cv_text_feature_responseCoding.T/cv_text_feature_responseCoding.sum(axis=1)).T

train_text_feature_onehotCoding=normalize(train_text_feature_onehotCoding,axis=0)
test_text_feature_onehotCoding=text_vectorizer.transform(test_df['TEXT'])
test_text_feature_onehotCoding=normalize(test_text_feature_onehotCoding,axis=0)
#cv_vectorizer=CountVectorizer(min_df=3)
cv_text_feature_onehotCoding=text_vectorizer.transform(cv_df['TEXT'])
cv_text_feature_onehotCoding=normalize(cv_text_feature_onehotCoding,axis=0)

sorted_text_fea_dict=dict(sorted(text_fea_dict.items(),key=lambda x: x[1], reverse=True))
sorted_text_occur=np.array(list(sorted_text_fea_dict.values()))
print(Counter(sorted_text_occur))

cv_log_error_array=[]

for i in alpha:
    clf=SGDClassifier(alpha=i,penalty='l2' ,loss='log',random_state=42)
    clf.fit(train_text_feature_onehotCoding,y_train)
    sig_clf=CalibratedClassifierCV(clf,method='sigmoid')
    sig_clf.fit(train_text_feature_onehotCoding,y_train)
    predict_y=sig_clf.predict_proba(cv_text_feature_onehotCoding)
    cv_log_error_array.append(log_loss(y_cv, predict_y,labels=clf.classes_,eps=1e-15))
    print('For alpha= ',i,' log loss is ',log_loss(y_cv,predict_y,labels=clf.classes_,eps=1e-15))

fig,ax=plt.subplots()
ax.plot(alpha,cv_log_error_array,c='g')
for i, txt in enumerate(np.round_(cv_log_error_array,3)):
    ax.annotate((alpha[i],np.round(txt,3)),(alpha[i],cv_log_error_array[i]))
plt.grid()
plt.title("cross validation")
plt.xlabel("alpha i")
plt.ylabel("error ")
plt.show()

best_alpha=np.argmin(cv_log_error_array)
clf=SGDClassifier(alpha=alpha[best_alpha],penalty='l2',loss='log',random_state=42)
clf.fit(train_text_feature_onehotCoding,y_train)
sig_clf=CalibratedClassifierCV(clf,method='sigmoid')
sig_clf.fit(train_text_feature_onehotCoding,y_train)

predict_y=sig_clf.predict_proba(train_text_feature_onehotCoding)
print('For values of best alpha ',alpha[best_alpha],' train log loss is ',log_loss(y_train,predict_y,labels=clf.classes_,eps=1e-15))

predict_y=sig_clf.predict_proba(cv_text_feature_onehotCoding)
print('For values of best alpha ',alpha[best_alpha],' cv log loss is ',log_loss(y_cv,predict_y,labels=clf.classes_,eps=1e-15))

predict_y=sig_clf.predict_proba(test_text_feature_onehotCoding)
print('For values of best alpha ',alpha[best_alpha],' test log loss is ',log_loss(y_test,predict_y,labels=clf.classes_,eps=1e-15))
                                   
def get_intersec_text(df):
    df_text_vec=CountVectorizer(min_df=3)
    df_text_fea=df_text_vec.fit_transform(df['TEXT'])
    df_text_features=df_text_vec.get_feature_names()

    df_text_fea_counts=df_text_fea.sum(axis=0).A1
    df_text_fea_dict=dict(zip(list(df_text_features),df_text_fea_counts))
    len1=len(set(df_text_features))
    len2=len(set(train_text_features) & set(df_text_features))
    return len1,len2

len1,len2=get_intersec_text(test_df)
print(np.round((len2/len1)*100,3),'% of word of test data are in train data')
len1,len2=get_intersec_text(cv_df)
print(np.round((len2/len1)*100,3),'% of word of cv data are in train data')

#data prepration
def report_log_loss(train_x,train_y,test_x,test_y,clf):
    clf.fit(train_x,train_y)
    sig_clf=CalibratedClassifierCV(clf,method='sigmoid')
    clf.fit(train_x,train_y)
    sif_clf_probs=sig_clf.predict_proba(test_x)
    return log_loss(test_y,sig_clf_probs,eps=1e-15)

def plot_confusion_matrix(test_y,predict_y):
    C=confusion_matrix(test_y,predict_y)
    labels=[1,2,3,4,5,6,7,8,9]
    plt.figure(figsize=(20,7))
    sns.heatmap(C,annot=True,cmap='YlGnBu',fmt=".3f",xticklabels=labels,yticklabels=labels)
    plt.xlabel('Predicted class')
    plt.ylabel('Original class')
    plt.show()

    #precision mattrix....colmn
    B=(C/C.sum(axis=0))
    plt.figure(figsize=(20,7))
    sns.heatmap(B,annot=True,cmap='YlGnBu',fmt=".3f",xticklabels=labels,yticklabels=labels)
    plt.xlabel('Predicted class')
    plt.ylabel('Original class')
    plt.show()

    #recall matrix...row

    A=(((C.T/C.sum(axis=1))).T)
    plt.figure(figsize=(20,7))
    sns.heatmap(A,annot=True,cmap='YlGnBu',fmt=".3f",xticklabels=labels,yticklabels=labels)
    plt.xlabel('Predicted class')
    plt.ylabel('Original class')
    plt.show()

def predict_and_plot_confusion_matrix(train_x,train_y,test_x,test_y,clf):
    clf.fit(train_x,train_y)
    sig_clf=CalibratedClassifierCV(clf,method="sigmoid")
    sig_clf.fit(train_x,train_y)
    pred_y=sig_clf.predict(test_x)

    print("log loss :",log_loss(test_y,sig_clf.predict_proba(test_x)))
    print("Number of mis-classified points :", np.count_nonzero((pred_y-test_y))/test_y.shape[0])
    plot_confusion_matrix(test_y,pred_y)


def get_impfeature_names(indices,text,gene,var,no_features):
    gene_count_vec=CountVectorizer()
    var_count_vec=CountVectorizer()
    text_count_vec=CountVectorizer(min_df=3)
    gene_vec=gene_count_vec.fit(train_df['Gene'])
    var_vec=var_count_vec.fit(train_df['Variation'])
    text_vec=text_count_vec.fit(train_df['TEXT'])
    feal_len=len(gene_vec.get_feature_names())
    fea2_len=len(var_count_vec.get_feature_names())

    word_present=0
    for i,v in enumerate(indices):
        if(v<fea1_len):
            word=gene_vec.get_feature_names()[v]
            yes_no=True if word==gene else False
            if yes_no:
                word_present+=1
                print(i,"gene feature[{}] present in test data point [{}]".format(word,yes_no))
        elif(v<fea1_len+fea2_len):
            word=var_vec.get_feature_names()[v-(fea1_len)]
            yes_no=True if word in text.split() else False
            if yes_no:
                 word_present+=1
                 print(i,"Variation feature[{}] present in test data point [{}]".format(word,yes_no))
                
        else:
            word=text_vec.get_feature_names()[v-(fea1_len+fea2_len)]
            yes_no=True if word in text.split() else False
            if yes_no:
                 word_present+=1
                 print(i,"Text feature[{}] present in test data point [{}]".format(word,yes_no))

    print("Out of top ",no_features,"features ",word_present,"are present in query point" )

train_gene_var_onehotCoding=hstack((train_gene_feature_onehotCoding,train_variation_feature_onehotCoding))
test_gene_var_onehotCoding=hstack((test_gene_feature_onehotCoding,test_variation_feature_onehotCoding))
cv_gene_var_onehotCoding=hstack((cv_gene_feature_onehotCoding,cv_variation_feature_onehotCoding))

train_x_onehotCoding=hstack((train_gene_var_onehotCoding,train_text_feature_onehotCoding)).tocsr()
train_y=np.array(list(train_df['Class']))

test_x_onehotCoding=hstack((test_gene_var_onehotCoding,test_text_feature_onehotCoding)).tocsr()
test_y=np.array(list(test_df['Class']))

cv_x_onehotCoding=hstack((cv_gene_var_onehotCoding,cv_text_feature_onehotCoding)).tocsr()
cv_y=np.array(list(cv_df['Class']))

train_gene_var_responseCoding=np.hstack((train_gene_feature_responseCoding,train_variation_feature_responseCoding))
test_gene_var_responseCoding=np.hstack((test_gene_feature_responseCoding,test_variation_feature_responseCoding))
cv_gene_var_responseCoding=np.hstack((cv_gene_feature_responseCoding,cv_variation_feature_responseCoding))

train_x_responseCoding=np.hstack((train_gene_var_responseCoding,train_text_feature_responseCoding))
#train_y=np.array(list(train_df['Class']))

test_x_responseCoding=np.hstack((test_gene_var_responseCoding,test_text_feature_responseCoding))

#test_y=np.array(list(test_df['Class']))

cv_x_responseCoding=np.hstack((cv_gene_var_responseCoding,cv_text_feature_responseCoding))
#cv_y=np.array(list(cv_df['Class']))

print("one hot encoding features: ")
print("(no of data points * no of features) in train data=" ,train_x_onehotCoding.shape)
print("(no of data points * no of features) in test data=" ,test_x_onehotCoding.shape)
print("(no of data points * no of features) in cv data=" ,cv_x_onehotCoding.shape)

print("response encoding features: ")
print("(no of data points * no of features) in train data=" ,train_x_responseCoding.shape)
print("(no of data points * no of features) in test data=" ,test_x_responseCoding.shape)
print("(no of data points * no of features) in cv data=" ,cv_x_responseCoding.shape)

#build machine learning models

#naive bayes......

alpha=[0.00001,0.0001,0.001,0.1,1,10,100,1000]
cv_log_error_array=[]

for i in alpha:
    print("for aplpha= ",i)
    clf=MultinomialNB(alpha=i)
    clf.fit(train_x_onehotCoding,train_y)
    sig_clf=CalibratedClassifierCV(clf,method='sigmoid')
    sig_clf.fit(train_x_onehotCoding,train_y)
    sig_clf_probs=sig_clf.predict_proba(cv_x_onehotCoding)
    cv_log_error_array.append(log_loss(cv_y, sig_clf_probs,labels=clf.classes_,eps=1e-15))
    print('For alpha= ',i,' log loss is ',log_loss(cv_y, sig_clf_probs,labels=clf.classes_,eps=1e-15))


fig,ax=plt.subplots()
ax.plot(np.log10(alpha),cv_log_error_array,c="g")
for i, txt in enumerate(np.round_(cv_log_error_array,3)):
    ax.annotate((alpha[i],str(txt)),(np.log10(alpha[i]),cv_log_error_array[i]))
plt.grid()
plt.xticks(np.log10(alpha))
plt.title("cross validation error for each alpha")
plt.xlabel("alpha is")
plt.ylabel("error ")
plt.show()

best_alpha=np.argmin(cv_log_error_array)
clf=MultinomialNB(alpha=alpha[best_alpha])
clf.fit(train_x_onehotCoding,train_y)
sig_clf=CalibratedClassifierCV(clf,method='sigmoid')
sig_clf.fit(train_x_onehotCoding,train_y)
#predict_y=sig_clf.predict_proba(cv_x_onehotCoding)

predict_y=sig_clf.predict_proba(train_x_onehotCoding)
print('For values of best alpha ',alpha[best_alpha],' train log loss is ',log_loss(train_y,predict_y,labels=clf.classes_,eps=1e-15))

predict_y=sig_clf.predict_proba(cv_x_onehotCoding)
print('For values of best alpha ',alpha[best_alpha],' cv log loss is ',log_loss(cv_y,predict_y,labels=clf.classes_,eps=1e-15))

predict_y=sig_clf.predict_proba(test_x_onehotCoding)
print('For values of best alpha ',alpha[best_alpha],' test log loss is ',log_loss(test_y,predict_y,labels=clf.classes_,eps=1e-15))

clf=MultinomialNB(alpha=alpha[best_alpha])
clf.fit(train_x_onehotCoding,train_y)
sig_clf=CalibratedClassifierCV(clf,method='sigmoid')
sig_clf.fit(train_x_onehotCoding,train_y)
#predict_y=sig_clf.predict_proba(cv_x_onehotCoding)

#sig_clf.fit(train_x_onehotCoding,train_y)
sig_clf_probs=sig_clf.predict_proba(cv_x_onehotCoding)
#cv_log_error_array.append(log_loss(cv_y, sig_clf_probs,labels=clf.classes_,eps=1e-15))
print('log loss is ',log_loss(cv_y, sig_clf_probs))

"""
def predict_and_plot_confusion_matrix(train_x,train_y,test_x,test_y,clf):
    clf.fit(train_x,train_y)
    sig_clf=CalibratedClassifierCV(clf,method="sigmoid")
    sig_clf.fit(train_x,train_y)
    pred_y=sif_clf.predict(text_x)

    print("log loss :",log_loss(test_y,sig_clf.predict_proba(text_x)))
    print("Number of mis-classified points :", np.count_nonzero((pred_y-test_y))/test_y.shape[0])
    plot_confusion_matrix(test_y,pred_y)
"""


clf=MultinomialNB(alpha=alpha[best_alpha])
predict_and_plot_confusion_matrix(train_x_onehotCoding,train_y,cv_x_onehotCoding,cv_y,clf)

print("Number of missclassified point :",np.count_nonzero((sig_clf.predict(cv_x_onehotCoding))-cv_y)/ cv_y.shape[0])
plot_confusion_matrix(cv_y,sig_clf.predict(cv_x_onehotCoding.toarray()))

#KNN classification

alpha=[5,11,15,21,31,41,51,99]
cv_log_error_array=[]

for i in alpha:
    print("for alpha= ",i)
    clf=KNeighborsClassifier(n_neighbors=i)
    clf.fit(train_x_responseCoding,train_y)
    sig_clf=CalibratedClassifierCV(clf,method='sigmoid')
    sig_clf.fit(train_x_responseCoding,train_y)
    sig_clf_probs=sig_clf.predict_proba(cv_x_responseCoding)
    cv_log_error_array.append(log_loss(cv_y, sig_clf_probs,labels=clf.classes_,eps=1e-15))
    print('For alpha= ',i,' log loss is ',log_loss(cv_y, sig_clf_probs,labels=clf.classes_,eps=1e-15))

fig,ax=plt.subplots()
ax.plot(np.log10(alpha),cv_log_error_array,c="g")
for i, txt in enumerate(np.round_(cv_log_error_array,3)):
    ax.annotate((alpha[i],str(txt)),(np.log10(alpha[i]),cv_log_error_array[i]))
plt.grid()
plt.xticks(np.log10(alpha))
plt.title("cross validation error for each alpha")
plt.xlabel("alpha is")
plt.ylabel("error ")
plt.show()

best_alpha=np.argmin(cv_log_error_array)
clf=KNeighborsClassifier(n_neighbors=alpha[best_alpha])
clf.fit(train_x_responseCoding,train_y)
sig_clf=CalibratedClassifierCV(clf,method='sigmoid')
sig_clf.fit(train_x_responseCoding,train_y)
#predict_y=sig_clf.predict_proba(cv_x_onehotCoding)

predict_y=sig_clf.predict_proba(train_x_responseCoding)
print('For values of best alpha ',alpha[best_alpha],' train log loss is ',log_loss(train_y,predict_y,labels=clf.classes_,eps=1e-15))

predict_y=sig_clf.predict_proba(cv_x_responseCoding)
print('For values of best alpha ',alpha[best_alpha],' cv log loss is ',log_loss(cv_y,predict_y,labels=clf.classes_,eps=1e-15))

predict_y=sig_clf.predict_proba(test_x_responseCoding)
print('For values of best alpha ',alpha[best_alpha],' test log loss is ',log_loss(test_y,predict_y,labels=clf.classes_,eps=1e-15))

clf=KNeighborsClassifier(n_neighbors=alpha[best_alpha])
predict_and_plot_confusion_matrix(train_x_responseCoding,train_y,cv_x_responseCoding,cv_y,clf)

clf=KNeighborsClassifier(n_neighbors=alpha[best_alpha])
clf.fit(train_x_responseCoding,train_y)
sig_clf=CalibratedClassifierCV(clf,method='sigmoid')
sig_clf.fit(train_x_responseCoding,train_y)

test_point_index=1
predicted_cls=sig_clf.predict(test_x_responseCoding[0].reshape(1,-1))
print(" Actual class :",predicted_cls[0])
print("Predicted class :",test_y[test_point_index])
neighbors=clf.kneighbors(test_x_responseCoding[test_point_index].reshape(1,-1))
print("the ",alpha[best_alpha],"nearest neighbors of test points belong to class",neighbors)
#print("Frequency of nearesr point :",Counter(train_y(neighbors[1][0])))


clf=KNeighborsClassifier(n_neighbors=alpha[best_alpha])
clf.fit(train_x_responseCoding,train_y)
sig_clf=CalibratedClassifierCV(clf,method='sigmoid')
sig_clf.fit(train_x_responseCoding,train_y)

test_point_index=100
predicted_cls=sig_clf.predict(test_x_responseCoding[0].reshape(1,-1))
print(" Actual class :",predicted_cls[0])
print("Predicted class :",test_y[test_point_index])
neighbors=clf.kneighbors(test_x_responseCoding[test_point_index].reshape(1,-1))
print("the ",alpha[best_alpha],"nearest neighbors of test points belong to class",neighbors)
#print("Frequency of nearesr point :",Counter(train_y(neighbors[1][0])))


                              
#Linear Regression
#balanced
alpha=[10 ** x for x in range(-6,3)]
cv_log_error_array=[]

for i in alpha:
    print("for alpha= ",i)
    clf=SGDClassifier(class_weight='balanced',alpha=i,penalty='l2' ,loss='log',random_state=42)
    clf.fit(train_x_onehotCoding,train_y)
    sig_clf=CalibratedClassifierCV(clf,method='sigmoid')
    sig_clf.fit(train_x_onehotCoding,train_y)
    sig_clf_probs=sig_clf.predict_proba(cv_x_onehotCoding)
    cv_log_error_array.append(log_loss(cv_y, sig_clf_probs,labels=clf.classes_,eps=1e-15))
    print('For alpha= ',i,' log loss is ',log_loss(cv_y, sig_clf_probs,labels=clf.classes_,eps=1e-15))

fig,ax=plt.subplots()
ax.plot(np.log10(alpha),cv_log_error_array,c="g")
for i, txt in enumerate(np.round_(cv_log_error_array,3)):
    ax.annotate((alpha[i],str(txt)),(np.log10(alpha[i]),cv_log_error_array[i]))
plt.grid()
plt.xticks(np.log10(alpha))
plt.title("cross validation error for each alpha")
plt.xlabel("alpha is")
plt.ylabel("error ")
plt.show()

best_alpha=np.argmin(cv_log_error_array)
clf=SGDClassifier(class_weight='balanced',alpha=alpha[best_alpha],penalty='l2' ,loss='log',random_state=42)
clf.fit(train_x_onehotCoding,train_y)
sig_clf=CalibratedClassifierCV(clf,method='sigmoid')
sig_clf.fit(train_x_onehotCoding,train_y)
#predict_y=sig_clf.predict_proba(cv_x_onehotCoding)

predict_y=sig_clf.predict_proba(train_x_onehotCoding)
print('For values of best alpha ',alpha[best_alpha],' train log loss is ',log_loss(train_y,predict_y,labels=clf.classes_,eps=1e-15))

predict_y=sig_clf.predict_proba(cv_x_onehotCoding)
print('For values of best alpha ',alpha[best_alpha],' cv log loss is ',log_loss(cv_y,predict_y,labels=clf.classes_,eps=1e-15))

predict_y=sig_clf.predict_proba(test_x_onehotCoding)
print('For values of best alpha ',alpha[best_alpha],' test log loss is ',log_loss(test_y,predict_y,labels=clf.classes_,eps=1e-15))

clf=SGDClassifier(class_weight='balanced',alpha=alpha[best_alpha],penalty='l2' ,loss='log',random_state=42)
predict_and_plot_confusion_matrix(train_x_onehotCoding,train_y,cv_x_onehotCoding,cv_y,clf)
"""
def get_imp_feature_names(text,indices,removed_ind=[]):
    word_present=0
    tabulate_list=[]
    incresingorder_ind=0
    for i in indices:
        if i<train_gene_feature_onehotcoding.shape[1]:
            tabulate_list.append([incresingorder_ind,'Gene','Yes'])
        elif i<18:
            tabulate_list.append([incresingorder_ind,'Variation','Yes'])
        if((i>17) & (i not in removed_ind)):
            word=train_text_features[i]
            yes_no=True if word in text.split() else False
            if yes_no:
                word_present+=1
            tabulate_list.append([incresingorder_ind,train_text_features[i],'Variation','Yes'])

clf=SGDClassifier(class_weight='balanced',alpha=alpha[best_alpha],penalty='l2' ,loss='log',random_state=42)
clf.fit(train_x_onehotCoding,train_y)
sig_clf=CalibratedClassifierCV(clf,method='sigmoid')
sig_clf.fit(train_x_onehotCoding,train_y)

test_point_index=1
predicted_cls=sig_clf.predict(test_x_onehotCoding[0].reshape(1,-1))
print(" Actual class :",predicted_cls[0])
print("Predicted class :",test_y[test_point_index])
neighbors=clf.kneighbors(test_x_onehoCoding[test_point_index].reshape(1,-1))
print("the ",alpha[best_alpha],"nearest neighbors of test points belong to class",neighbors)
print("Frequency of nearesr point :",Counter(train_y(neighbors[1][0])))


clf=SGDClassifier(class_weight='balanced',alpha=alpha[best_alpha],penalty='l2' ,loss='log',random_state=42)
clf.fit(train_x_onehotCoding,train_y)
sig_clf=CalibratedClassifierCV(clf,method='sigmoid')
sig_clf.fit(train_x_onehotCoding,train_y)

test_point_index=100
predicted_cls=sig_clf.predict(test_x_onehotCoding[0].reshape(1,-1))
print(" Actual class :",predicted_cls[0])
print("Predicted class :",test_y[test_point_index])
neighbors=clf.kneighbors(test_x_onehotCoding[test_point_index].reshape(1,-1))
print("the ",alpha[best_alpha],"nearest neighbors of test points belong to class",neighbors)
print("Frequency of nearesr point :",Counter(train_y(neighbors[1][0])))
     """ 
#linear regression without balancing


alpha=[10 ** x for x in range(-6,3)]
cv_log_error_array=[]

for i in alpha:
    print("for alpha= ",i)
    clf=SGDClassifier(alpha=i,penalty='l2' ,loss='log',random_state=42)
    clf.fit(train_x_onehotCoding,train_y)
    sig_clf=CalibratedClassifierCV(clf,method='sigmoid')
    sig_clf.fit(train_x_onehotCoding,train_y)
    sig_clf_probs=sig_clf.predict_proba(cv_x_onehotCoding)
    cv_log_error_array.append(log_loss(cv_y, sig_clf_probs,labels=clf.classes_,eps=1e-15))
    print('For alpha= ',i,' log loss is ',log_loss(cv_y, sig_clf_probs,labels=clf.classes_,eps=1e-15))

fig,ax=plt.subplots()
ax.plot(np.log10(alpha),cv_log_error_array,c="g")
for i, txt in enumerate(np.round_(cv_log_error_array,3)):
    ax.annotate((alpha[i],str(txt)),(np.log10(alpha[i]),cv_log_error_array[i]))
plt.grid()
plt.xticks(np.log10(alpha))
plt.title("cross validation error for each alpha")
plt.xlabel("alpha is")
plt.ylabel("error ")
plt.show()

best_alpha=np.argmin(cv_log_error_array)
clf=SGDClassifier(alpha=alpha[best_alpha],penalty='l2' ,loss='log',random_state=42)
clf.fit(train_x_onehotCoding,train_y)
sig_clf=CalibratedClassifierCV(clf,method='sigmoid')
sig_clf.fit(train_x_onehotCoding,train_y)
#predict_y=sig_clf.predict_proba(cv_x_onehotCoding)

predict_y=sig_clf.predict_proba(train_x_onehotCoding)
print('For values of best alpha ',alpha[best_alpha],' train log loss is ',log_loss(train_y,predict_y,labels=clf.classes_,eps=1e-15))

predict_y=sig_clf.predict_proba(cv_x_onehotCoding)
print('For values of best alpha ',alpha[best_alpha],' cv log loss is ',log_loss(cv_y,predict_y,labels=clf.classes_,eps=1e-15))

predict_y=sig_clf.predict_proba(test_x_onehotCoding)
print('For values of best alpha ',alpha[best_alpha],' test log loss is ',log_loss(test_y,predict_y,labels=clf.classes_,eps=1e-15))

clf=SGDClassifier(class_weight='balanced',alpha=alpha[best_alpha],penalty='l2' ,loss='log',random_state=42)
predict_and_plot_confusion_matrix(train_x_onehotCoding,train_y,cv_x_onehotCoding,cv_y,clf)

#linear support vector machines
alpha=[10 ** x for x in range(-5,3)]
cv_log_error_array=[]

for i in alpha:
    print("for alpha= ",i)
    clf=SGDClassifier(class_weight='balanced',alpha=i,penalty='l2' ,loss='log',random_state=42)
    clf.fit(train_x_onehotCoding,train_y)
    sig_clf=CalibratedClassifierCV(clf,method='sigmoid')
    sig_clf.fit(train_x_onehotCoding,train_y)
    sig_clf_probs=sig_clf.predict_proba(cv_x_onehotCoding)
    cv_log_error_array.append(log_loss(cv_y, sig_clf_probs,labels=clf.classes_,eps=1e-15))
    print('For alpha= ',i,' log loss is ',log_loss(cv_y, sig_clf_probs,labels=clf.classes_,eps=1e-15))

fig,ax=plt.subplots()
ax.plot(np.log10(alpha),cv_log_error_array,c="g")
for i, txt in enumerate(np.round_(cv_log_error_array,3)):
    ax.annotate((alpha[i],str(txt)),(np.log10(alpha[i]),cv_log_error_array[i]))
plt.grid()
plt.xticks(np.log10(alpha))
plt.title("cross validation error for each alpha")
plt.xlabel("alpha is")
plt.ylabel("error ")
plt.show()

best_alpha=np.argmin(cv_log_error_array)
clf=SGDClassifier(class_weight='balanced',alpha=alpha[best_alpha],penalty='l2' ,loss='log',random_state=42)
clf.fit(train_x_onehotCoding,train_y)
sig_clf=CalibratedClassifierCV(clf,method='sigmoid')
sig_clf.fit(train_x_onehotCoding,train_y)
#predict_y=sig_clf.predict_proba(cv_x_onehotCoding)

predict_y=sig_clf.predict_proba(train_x_onehotCoding)
print('For values of best alpha ',alpha[best_alpha],' train log loss is ',log_loss(train_y,predict_y,labels=clf.classes_,eps=1e-15))

predict_y=sig_clf.predict_proba(cv_x_onehotCoding)
print('For values of best alpha ',alpha[best_alpha],' cv log loss is ',log_loss(cv_y,predict_y,labels=clf.classes_,eps=1e-15))

predict_y=sig_clf.predict_proba(test_x_onehotCoding)
print('For values of best alpha ',alpha[best_alpha],' test log loss is ',log_loss(test_y,predict_y,labels=clf.classes_,eps=1e-15))

clf=SGDClassifier(class_weight='balanced',alpha=alpha[best_alpha],penalty='l2' ,loss='log',random_state=42)
predict_and_plot_confusion_matrix(train_x_onehotCoding,train_y,cv_x_onehotCoding,cv_y,clf)


#random forest for onehotCoding

alpha=[100,200,500,1000,2000]
max_depth=[5,10]
cv_log_error_array=[]

for i in alpha:
    for j in max_depth:
        print("for alpha= ",i,' j is ',j)
        clf=RandomForestClassifier(n_estimators=i,criterion='gini',max_depth=j,loss='log',random_state=42)
        clf.fit(train_x_onehotCoding,train_y)
        sig_clf=CalibratedClassifierCV(clf,method='sigmoid')
        sig_clf.fit(train_x_onehotCoding,train_y)
        sig_clf_probs=sig_clf.predict_proba(cv_x_onehotCoding)
        cv_log_error_array.append(log_loss(cv_y, sig_clf_probs,labels=clf.classes_,eps=1e-15))
        print('For alpha= ',i,' log loss is ',log_loss(cv_y, sig_clf_probs,labels=clf.classes_,eps=1e-15))

fig,ax=plt.subplots()
ax.plot(np.log10(alpha),cv_log_error_array,c="g")
for i, txt in enumerate(np.round_(cv_log_error_array,3)):
    ax.annotate((alpha[i],str(txt)),(np.log10(alpha[i]),cv_log_error_array[i]))
plt.grid()
plt.xticks(np.log10(alpha))
plt.title("cross validation error for each alpha")
plt.xlabel("alpha is")
plt.ylabel("error ")
plt.show()

best_alpha=np.argmin(cv_log_error_array)
clf=clf=RandomForestClassifier(n_estimators=alpha[int(best_alpha/2)],criterion='gini',max_depth=j,penalty='l2' ,loss='log',random_state=42)
clf.fit(train_x_onehotCoding,train_y)
sig_clf=CalibratedClassifierCV(clf,method='sigmoid')
sig_clf.fit(train_x_onehotCoding,train_y)
#predict_y=sig_clf.predict_proba(cv_x_onehotCoding)

predict_y=sig_clf.predict_proba(train_x_onehotCoding)
print('For values of best alpha ',alpha[best_alpha],' train log loss is ',log_loss(train_y,predict_y,labels=clf.classes_,eps=1e-15))

predict_y=sig_clf.predict_proba(cv_x_onehotCoding)
print('For values of best alpha ',alpha[best_alpha],' cv log loss is ',log_loss(cv_y,predict_y,labels=clf.classes_,eps=1e-15))

predict_y=sig_clf.predict_proba(test_x_onehotCoding)
print('For values of best alpha ',alpha[best_alpha],' test log loss is ',log_loss(test_y,predict_y,labels=clf.classes_,eps=1e-15))

clf=RandomForestClassifier(n_estimators=i,criterion='gini',max_depth=j,penalty='l2' ,loss='log',random_state=42)
predict_and_plot_confusion_matrix(train_x_onehotCoding,train_y,cv_x_onehotCoding,cv_y,clf)

#random forest for responseCoding

alpha=[100,200,500,1000,2000]
max_depth=[2,3,5,10]
cv_log_error_array=[]

for i in alpha:
    for j in max_depth:
        print("for alpha= ",i,' j is ',j)
        clf=RandomForestClassifier(n_estimators=i,criterion='gini',max_depth=j,penalty='l2' ,loss='log',random_state=42)
        clf.fit(train_x_responseCoding,train_y)
        sig_clf=CalibratedClassifierCV(clf,method='sigmoid')
        sig_clf.fit(train_x_responseCoding,train_y)
        sig_clf_probs=sig_clf.predict_proba(cv_x_responseCoding)
        cv_log_error_array.append(log_loss(cv_y, sig_clf_probs,labels=clf.classes_,eps=1e-15))
        print('For alpha= ',i,' log loss is ',log_loss(cv_y, sig_clf_probs,labels=clf.classes_,eps=1e-15))

fig,ax=plt.subplots()
ax.plot(np.log10(alpha),cv_log_error_array,c="g")
for i, txt in enumerate(np.round_(cv_log_error_array,3)):
    ax.annotate((alpha[i],str(txt)),(np.log10(alpha[i]),cv_log_error_array[i]))
plt.grid()
plt.xticks(np.log10(alpha))
plt.title("cross validation error for each alpha")
plt.xlabel("alpha is")
plt.ylabel("error ")
plt.show()

best_alpha=np.argmin(cv_log_error_array)
clf=clf=RandomForestClassifier(n_estimators=alpha[int(best_alpha/2)],criterion='gini',max_depth=j,penalty='l2' ,loss='log',random_state=42)
clf.fit(train_x_responseCoding,train_y)
sig_clf=CalibratedClassifierCV(clf,method='sigmoid')
sig_clf.fit(train_x_responseCoding,train_y)
#predict_y=sig_clf.predict_proba(cv_x_onehotCoding)

predict_y=sig_clf.predict_proba(train_x_responseCoding)
print('For values of best alpha ',alpha[best_alpha],' train log loss is ',log_loss(train_y,predict_y,labels=clf.classes_,eps=1e-15))

predict_y=sig_clf.predict_proba(cv_x_responseCoding)
print('For values of best alpha ',alpha[best_alpha],' cv log loss is ',log_loss(cv_y,predict_y,labels=clf.classes_,eps=1e-15))

predict_y=sig_clf.predict_proba(test_x_responseCoding)
print('For values of best alpha ',alpha[best_alpha],' test log loss is ',log_loss(test_y,predict_y,labels=clf.classes_,eps=1e-15))

clf=RandomForestClassifier(n_estimators=i,criterion='gini',max_depth=j,penalty='l2' ,loss='log',random_state=42)
predict_and_plot_confusion_matrix(train_x_responseCoding,train_y,cv_x_responseCoding,cv_y,clf)

