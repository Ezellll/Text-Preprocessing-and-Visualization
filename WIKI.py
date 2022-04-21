# Metin Ön işleme ve Görselleştirme

################
# Problem
################

# Wikipedia metinleri içeren veri setine metin ön işleme ve görselleştirme yapınız

##############################
# Proje Görevleri
###############################

from warnings import filterwarnings
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from textblob import Word, TextBlob
from wordcloud import WordCloud

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


############################################################
# Görev 1: Metin Ön İşleme İşlemlerini Gerçekleştiriniz
############################################################


df = pd.read_csv("Homework/wiki_data.csv")
df.head()


# Adım 1: Metin ön işleme için clean_text adında fonksiyon oluşturunuz. Fonksiyon;
# • Büyük küçük harf dönüşümü,
# • Noktalama işaretlerini çıkarma,
# • Numerik ifadeleri çıkarma Işlemlerini gerçekleştirmeli.
df["text"].str.replace('\d', '')
def clean_text(DataFrame,col):
    #Büyük küçük harf dönüşümü,
    DataFrame[col] = DataFrame[col].str.lower()
    # Noktalama işaretlerini çıkarma,
    DataFrame[col] = DataFrame[col].str.replace('[^\w\s]', '')
    # Numerik ifadeleri çıkarma Işlemlerini gerçekleştirmeli.
    DataFrame[col] = df[col].str.replace('\d', '')

# Adım 2: Yazdığınız fonksiyonu veri seti içerisindeki tüm metinlere uygulayınız.

clean_text(df, "text")
df.head()


# Adım 3: Metin içinde öznitelik çıkarımı yaparken önemli olmayan kelimeleri (ben, sen, de, da, ki, ile vs) çıkaracak remove_stopwords adında
# fonksiyon yazınız.

import nltk
nltk.download('stopwords')

def remove_stopwords(DataFrame,col,language="english"):
    sw = stopwords.words(language)
    DataFrame[col] = DataFrame[col].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))


# Adım 4: Yazdığınız fonksiyonu veri seti içerisindeki tüm metinlere uygulayınız.

remove_stopwords(df, "text")
df.head()


# Adım 5: Metinde az geçen (1000'den az, 2000'den az gibi) kelimeleri bulunuz. Ve bu kelimeleri metin içerisinden çıkartınız.

temp_df = pd.Series(" ".join(df["text"]).split()).value_counts()
# 1000'den az tekrar eden kelimeleri filtreledim.
drops = temp_df[temp_df < 1000]
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))

"asdfsd343sdasd".replace("\d","")

# Adım 6: Metinleri tokenize edip sonuçları gözlemleyiniz.

#Cümlelerini parçalarına ayırma

#nltk.download("punkt")
df["text"].apply(lambda x: TextBlob(x).words).head()

# Adım 7: Lemmatization işlemi yapınız

# Kelimeleri köklerine ayırma

nltk.download('wordnet')
df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


##############################################################
#Görev 2: Veriyi Görselleştiriniz (Text Visualization)
##############################################################

# Adım 1: Metindeki terimlerin frekanslarını hesaplayınız.


tf = df["text"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ["words", "tf"]
tf.sort_values("tf", ascending=False)

# Adım 2: Bir önceki adımda bulduğunuz terim frekanslarının Barplot grafiğini oluşturunuz.


# Ürün arama sürecinde kullanılabilir...
tf[tf["tf"] > 7000].plot.bar(x="words", y="tf")
plt.show()

# Adım 3: Kelimeleri WordCloud ile görselleştiriniz

text = " ".join(i for i in df.text)
# Kelime bulutu oluşturma
wordcloud = WordCloud().generate(text)
#Görselleştirme
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Görselleştirmede biçimlendirme yapmak
wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="green").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Görseli kaydetme
wordcloud.to_file("wordcloud_wiki_hw.png")



#############################################################
# Görev 3: Tüm Aşamaları Tek Bir Fonksiyon Olarak Yazınız
##############################################################

# Adım 1: Metin ön işleme işlemlerini gerçekleştiriniz.

def text_prep_visulation(DataFrame,col,rare_word_th = 1,Barplot=False,Wordcloud=False):
    # Remove Punctuations and Numbers
    DataFrame = clean_text(DataFrame, col)
    #Stopwords
    DataFrame = remove_stopwords(DataFrame, col)
    #Rarewords
    temp_df = pd.Series(" ".join(DataFrame[col]).split()).value_counts()
    drops = temp_df[temp_df < rare_word_th]
    DataFrame[col] = DataFrame[col].apply(lambda x: " ".join(x for x in x.split() if x not in drops))
    #Tokenization
    DataFrame[col].apply(lambda x: TextBlob(x).words).head()
    #Lemmatization
    DataFrame[col] = DataFrame[col].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

# Adım 2: Görselleştirme işlemlerini fonksiyona argüman olarak ekleyiniz.

def text_prep_visulation(DataFrame,col,rare_word_th = 1,Barplot=False,Wordcloud=False):
    #Punctuations and Numbers
    DataFrame = clean_text(DataFrame, col)
    #Stopwords
    DataFrame = remove_stopwords(DataFrame, col)
    #Rarewords
    temp_df = pd.Series(" ".join(DataFrame[col]).split()).value_counts()
    # rare_word_th'a göre tekrar eden kelimeleri filtreliyeceğiz
    drops = temp_df[temp_df < rare_word_th]
    DataFrame[col] = DataFrame[col].apply(lambda x: " ".join(x for x in x.split() if x not in drops))
    #Tokenization
    DataFrame[col].apply(lambda x: TextBlob(x).words).head()
    #Lemmatization
    DataFrame[col] = DataFrame[col].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

    if Barplot:
        #Barplot(sütüngrafiği)
        tf = DataFrame[col].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
        tf.columns = ["words", "tf"]
        tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
        plt.show()
    if Wordcloud:
        # Wordcloud(Kelime Bulutu)
        text = " ".join(i for i in df.reviewText)
        wordcloud = WordCloud().generate(text)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()


# Adım 3: Fonksiyonu açıklayan 'docstring' yazınız.

def text_prep_visulation(DataFrame,col,rare_word_th = 1,Barplot=False,Wordcloud=False):
    """
    Veri seti içerisindeki text bilgisi içeren kolonlara aşağıdaki
    metin ön işleme adımlarını uygular ve  Barplot, Wordcloud ile metin görselleştirme yapar.
    - Noktalama işaretlerini ve sayıları kaldırır,
    - Bilgisayar dilinde, etkisiz kelimeler kaldırılır (Remove Stopwords),
    -  Text içerisinde geçen kelimleri frekanslarına göre filtreler,
    - Tokenization işlemini uygular,
    - Lemmatization işlemini uygular.
    Args:
        DataFrame: pandas dataframe

        col: string
            Text bilgisini içeren kolon.
        rare_word_th: int
            Text içerisinde nadir geçen kelimlere filtreleme uygulamak için sınır değeri belirler.
        Barplot: bool
            Kelime frekanslarının görselleştirilmesini sütün grafiği ile sağlar.
        Wordcloud: bool
            Kelime frekanslarının görselleştirilmesini kelime bulutu ile sağlar.

    Returns:

    Examples
    -------
        df = pd.read_csv("Homework/wiki_data.csv")
        text_prep_visulation(df, "text",rare_word_th=1000, Barplot=True)
    """
    #Punctuations and Numbers
    clean_text(DataFrame, col)
    #Stopwords
    remove_stopwords(DataFrame, col)
    #Rarewords
    temp_df = pd.Series(" ".join(DataFrame[col]).split()).value_counts()
    # rare_word_th'a göre tekrar eden kelimeleri filtreliyeceğiz
    drops = temp_df[temp_df < rare_word_th]
    DataFrame[col] = DataFrame[col].apply(lambda x: " ".join(x for x in x.split() if x not in drops))
    #Tokenization
    DataFrame[col].apply(lambda x: TextBlob(x).words).head()
    #Lemmatization
    DataFrame[col] = DataFrame[col].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

    if Barplot:
        #Barplot(sütüngrafiği)
        tf = DataFrame[col].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
        tf.columns = ["words", "tf"]
        tf[tf["tf"] > 7000].plot.bar(x="words", y="tf")
        plt.show()
    if Wordcloud:
        # Wordcloud(Kelime Bulutu)
        text = " ".join(i for i in df.reviewText)
        wordcloud = WordCloud().generate(text)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()


df = pd.read_csv("Homework/wiki_data.csv")
text_prep_visulation(df, "text",rare_word_th=1000, Barplot=True)
df.head()
