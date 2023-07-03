import streamlit as st
import pandas as pd
from PIL import Image
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing import sequence
from files import text_cleaner
from deep_translator import GoogleTranslator
#--------------
@st.cache_data  
def load_data():
    df1=pd.read_csv('files/data_Foody.csv',index_col=0)
    df2=pd.read_csv('files/FoodyFinal.csv',index_col=0)
    return df1,df2
@st.cache_data
def load_file():
    ##LOAD EMOJICON
    file = open('files/emojicon.txt', 'r', encoding="utf8")
    emoji_lst = file.read().split('\n')
    emoji_dict = {}
    for line in emoji_lst:
        key, value = line.split('\t')
        emoji_dict[key] = str(value)
    file.close()
    #################
    #LOAD TEENCODE
    file = open('files/teencode.txt', 'r', encoding="utf8")
    teen_lst = file.read().split('\n')
    teen_dict = {}
    for line in teen_lst:
        key, value = line.split('\t')
        teen_dict[key] = str(value)
    file.close()
    ###############
    #LOAD TRANSLATE ENGLISH -> VNMESE
    file = open('files/english-vnmese.txt', 'r', encoding="utf8")
    english_lst = file.read().split('\n')
    english_dict = {}
    for line in english_lst:
        key, value = line.split('\t')
        english_dict[key] = str(value)
    file.close()
    ################
    # #LOAD wrong words
    # file = open('files/wrong-word.txt', 'r', encoding="utf8")
    # wrong_lst = file.read().split('\n')
    # file.close()
    file = open('files/vietnamese-stopwords_new.txt', 'r', encoding="utf8")
    stopwords_lst = file.read().split('\n')
    file.close()
    return (emoji_dict, teen_dict,english_lst,stopwords_lst)
def textprocessing(text,emoji_dict, teen_dict,english_lst,stopwords_lst):
    text=text_cleaner.process_text(text,emoji_dict, teen_dict, [], english_lst)
    text=text_cleaner.covert_unicode(text)
    text=text_cleaner.process_postag_thesea(text)
    text=text_cleaner.remove_stopword(text,stopwords_lst)
    return (text)
@st.cache_resource
def load_model():
    # from tensorflow.keras.models import load_model
    json_file = open("./Model/DeepLearning2ClassSmote.json" ,'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loadmodel = tf.keras.models.model_from_json(loaded_model_json)
    loadmodel.load_weights("./Model/DeepLearning2ClassSmote_weights.h5")
    # # loading
    with open('./Model/tokenizer2ClassSmote.pickle', 'rb') as handle:
        tok = pickle.load(handle)
    return (loadmodel,tok)
def CNNPredict(text):
    X_new = pd.Series(data = text)
    test_sequences_new = tok.texts_to_sequences(X_new)
    test_sequences_matrix_new = sequence.pad_sequences(test_sequences_new,maxlen=220)
    predict=loadmodel.predict(test_sequences_matrix_new,verbose=0)
    if predict[0][0]>=0.5:
        return 1
    else:
        return 0
# GUI
st.title("Data Science Project")
st.write("# Sentiment Analysis")
menu = ["1. Introduction","2. Model", "3. Prediction"]
choice = st.sidebar.selectbox('Menu', menu)
if choice == '1. Introduction':
    st.write("## 1. Introduction") 
    st.write("#### Business Objective/Problem")
    col1,col2=st.columns([1,3])
    with col1:
        image = Image.open('files/logo.png')
        st.image(image)
    with col2:
        st.write("""- Foody is a channel to cooperate with restaurants/eateries selling food online.
- We can go here to see reviews, comments, and order food.
- From customer reviews, the problem is how to make restaurants/eateries understand customers better and know how they evaluate themselves to improve more in service/products.""")
    st.info("""Based on the history of previous customer reviews => Data collected from customer comments and reviews at https://www.foody.vn/   
=> Objective/problem:Building a predictive model that helps restaurants know the quick feedback of customers about their products or services (positive, negative, or neutral), which helps The restaurant understands the business situation, understands the opinions of the customers, thereby allowing the restaurant to improve its services and products.""")
    st.write("#### Requirement of the problem")
    st.info("""- Text preprocessing
- Applying RNN, LSTM to predict
- Applying PhoBERT to predict""")
elif choice=="2. Model":
    sub_category = st.sidebar.radio("Sub category",('2.1 Text preprocessing','2.2 RNN, LSTM', '2.3 PhoBERT'))
    st.write("## 2. Model")
    df1,df2=load_data()
    if sub_category=="2.1 Text preprocessing": 
        st.subheader("2.1 Text preprocessing")  
        #------------------------------------------------------------------------------------------------------
        st.write("#### Data info")
        with st.expander("Raw Data:"):
            st.dataframe(df1)
        st.code("""Step 1: Raw data preprocessing
- Remove special characters
- Replace emojicon, teen code with corresponding text
- Replace some punctuation and numbers with spaces
- Replace misspelled words with spaces
Step 2: Standardize Vietnamese Unicode
Step 3: Tokenizer Vietnamese text using the library underthesea
Step 4: Delete Vietnamese stopwords""")
        st.write("#### Cleaned Data")
        with st.expander("Cleaned Data:"):
            st.dataframe(df2[["restaurant","review_text","CONTENT","review_score","CLASS"]])
        image = Image.open('files/Class.png')
        st.image(image)
        st.write("#### Visualization for each class")
        st.info('Class 0: Negative')
        col3,col4=st.columns([1,2])
        with col3:
            st.code("""Words that appear most
- được
- nhiều
- mua
- ngon
- không_có
- khác
- xong
- phục_vụ
- kêu
- tệ""")
        with col4:
            image = Image.open('files/Class0.png')
            st.image(image)
        st.info('Class 1: Positive')
        col3,col4=st.columns([1,2])
        with col3:
            st.code("""Words that appear most
- nhiều
- ngon
- được
- thích
- lắm
- thử
- khác
- ổn
- cười
- nướng""")
        with col4:
            image = Image.open('files/Class1.png')
            st.image(image)
        st.success("""Comment:   
If the data is divided into 3 classes, it will be easy to confuse when determining the neutral class    
=> Divide the data into 2 classes""")
    elif sub_category=="2.2 RNN, LSTM": 
        st.subheader("2.2 RNN, LSTM")  
        #------------------------------------------------------------------------------------------------------
        st.write("#### Build model")
        st.markdown("---")
        col8,col9= st.columns(2)
        with col8:
            st.info('Model 1')
            st.code("""Using Tokenizer
model=Sequential()
model.add(Embedding(max_words,256))
model.add(LSTM(128))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))""")
            st.write("Result:")
            image = Image.open('files/deeplearning.png')
            st.image(image)
            image = Image.open('files/report.JPG')
            st.image(image)
        with col9:
            st.info('Model 1 (SMOTE)')
            st.code("""Using Tokenizer
model=Sequential()
model.add(Embedding(max_words,256))
model.add(LSTM(128))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))""")
            st.write("Result:")
            image = Image.open('files/deeplearning2.png')
            st.image(image)
            image = Image.open('files/report2.JPG')
            st.image(image)
        st.markdown("---")
        st.write("#### Conclusion")
        st.success("The model which uses SMOTE has the best result")
    elif sub_category=="2.3 PhoBERT": 
        st.subheader("2.3 PhoBERT")  
        #------------------------------------------------------------------------------------------------------
        st.write("#### Build model")
        st.markdown("---")
        st.code("""Using PhoBERT to extract more features from the data
Building with these models: 
- SVC
- LinearSVC
- NuSVC
- LogisticRegression
- Perceptron
- CNN
""")
        st.write("Result:")
        col5,col6=st.columns(2)
        with col5:
            st.info('SVC')
            image = Image.open('files/SVC.JPG')
            st.image(image)
            st.info('LinearSVC')
            image = Image.open('files/LinearSVC.JPG')
            st.image(image)
            st.info('NuSVC')
            image = Image.open('files/NuSVC.JPG')
            st.image(image)
        with col6:
            st.info('LogisticRegression')
            image = Image.open('files/LogisticRegression.JPG')
            st.image(image)
            st.info('Perceptron')
            image = Image.open('files/Perceptron.JPG')
            st.image(image)
            st.info('CNN')
            image = Image.open('files/CNN.JPG')
            st.image(image)
        st.markdown("---")
        st.write("#### Conclusion")
        st.success("""NuSVC has the highest accuracy, but models using phoBERT have a different size (>500MB)
=> Only use RNN-LSTM for the prediction part""")
elif choice=="3. Prediction":
    emoji_dict, teen_dict,english_lst,stopwords_lst=load_file()
    loadmodel,tok=load_model()#
    st.write("## 3. Prediction")
    genre1 = st.radio(
        "Select prediction type",
        ('Input text (English)','Input text (Vietnamese)','Upload file'))
    if genre1=='Input text (English)':
        title = st.text_input('Type a comment (English)', '')
        if title.replace(" ","")=="":
            st.warning('Please type something, the comment cannot be null!', icon="⚠️")
        else:
            translator = GoogleTranslator(source='en', target='vi')
            text=translator.translate(title)
            #st.write(text)
            text=textprocessing(text,emoji_dict, teen_dict,english_lst,stopwords_lst)
            ### 2 CLASS
            k=CNNPredict(text)
            if k==1:
                st.success("Positive",icon="😜")
            else:
                st.error("Negative",icon="😡")
    elif genre1=='Input text (Vietnamese)':
        title = st.text_input('Type a comment (Vietnamese)', '')
        if title.replace(" ","")=="":
            st.warning('Please type something, the comment cannot be null!', icon="⚠️")
        else:
            text=textprocessing(title,emoji_dict, teen_dict,english_lst,stopwords_lst)
            ### 2 CLASS
            k=CNNPredict(text)
            if k==1:
                st.success("Positive",icon="😜")
            else:
                st.error("Negative",icon="😡")
    elif genre1=='Upload file':
        df2 = st.file_uploader("Upload file ", type={"csv"})
        if df2 is not None:
            try:
                df2 = pd.read_csv(df2)
                st.dataframe(df2)
                ok=0
                try: 
                    data=df2.review_text
                    ok=1
                except:
                    st.error('review_text column not found in file!', icon="🚨") 
                if ok==1:
                    df=df2.copy()
                    df["result"]=df.review_text.map(lambda x: CNNPredict(textprocessing(x,emoji_dict, teen_dict,english_lst,stopwords_lst)))
                    df["result"]=df["result"].map(lambda x: "Positive" if x==1 else "Negative")
                    st.write("Result:")
                    st.dataframe(df)
                    st.download_button("Download result",df.to_csv(index=False).encode("utf-8"),"result.csv","text/csv",key='download-csv') 
            except:
                st.error('Can not load this file!', icon="🚨") 