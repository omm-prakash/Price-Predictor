from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
cv = CountVectorizer(max_features=50)
from crypt import methods
from flask import request
from flask import render_template
from flask import current_app as app
import pickle
import pandas as pd
import numpy as np

##############################################################################################
laptopModel = pickle.load(open('pickle/laptopModel.pkl','rb'))   
laptops = pickle.load(open('pickle/laptops.pkl','rb'))   
similar_laptops = pickle.load(open('pickle/similar_laptops.pkl','rb'))   
X = pickle.load(open('pickle/X.pkl','rb'))   
###########################################################################

data={}
for col in laptops.columns:
    data[col] = list(laptops[col].unique())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/laptop')
def laptop():
    return render_template('laptop.html', data=data)

@app.route('/contact')
def contact():
    return render_template('contact.html')

bins = [1000, 10000, 25000, 50000, 75000, 100000, 125000, 175000, 225000, 275000, 325000, 375000, np.inf]
names = ['<10k', '10k-25k', '25k-50k', '50k-75k', '75k-100k', '100k-125k', '125k-175k', '175k-225k', '225k-275k', '275k-325k', '325k-375k', '375k+']

def similar_laptop(data):
  dic = {}
  for i,col in enumerate(similar_laptops.columns[2:]):
    dic[col]=data[i]
  dic['Company']='a'
  dic['Product']='b'
  temp1=similar_laptops.copy()
  temp1=temp1.append(dic,ignore_index=True)
  temp=pd.DataFrame()
  temp['comapany-product'] = temp1.Company +' '+ temp1.Product
  temp['tags'] = temp1.drop(['Company','Product'],axis=1).apply(lambda x: ' '.join(x),axis=1)

  vector = cv.fit_transform(temp['tags']).toarray()
  laptop_similar=cosine_similarity(vector)  
  print(laptop_similar.shape)
  lap=sorted(enumerate(laptop_similar[-1]),key=lambda x:x[1],reverse=True)[1:5]
  details=[]
  for bk in lap:
    details.append(temp.loc[bk[0],:])

  return details

# similar_laptop(['Ultrabook', 'Intel-i5', '8','2560x1600', '100k-125k', 'macOS', 'Retina-Display'])
# https://www.flipkart.com/search?q=ops&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&as=off
@app.route('/laptop_filter', methods=['POST'])
def laptop_filter():
    data = {}
    for col in X.columns:
        if col not in ['Touchscreen','IPS Panel']:
            data[col] = request.form[col]
        else:
            if request.form.get(col, False):
                data[col]=1 
            else:
                data[col]=0

    df = pd.DataFrame(data)
    print(df)
    price = pipe.predict(df)
    return price
