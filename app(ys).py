from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import streamlit as st
import pandas as pd 
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler

def main():
    st.title('K-Means 클러스터링 앱 ')

## 1.유저가 클러스터링 하고싶은 csv파일을 업로드

    file = st.file_uploader('파일 업로드', type=['csv'])

##2.업로드한 csv파일을 데이터프레임으로 읽고
    if file is not None :
        df = pd.read_csv(file)
        st.dataframe( df )

        # 결측값(nan) 처리한다.
        df.dropna(inplace=True)

        ##4.wcss를 확인하기 위한, 그룹의 갯수를 정할수있다.(1~10개)
        columns = df.columns
        selected_columns = st.multiselect('원하는 컬럼을 선택하세요' , columns )
        


        if len(selected_columns) != 0 :

            X = df[selected_columns]
            st.dataframe(X)

            # 문자열이 들어있으면 처리한 후에 화면 보여주기
            X_new = pd.DataFrame()

            for name in X.columns :
                print(name)
                

                
                #각컬럼 데이터를 가저온다
                data = X[name]
                data.rest_index(inplace=True, drop=True)

                # 문자열인지 아닌지 나눠서 처리하면 된다.
                if data.dtype == object :
                
                    #문자열이니까, 갯수가 2개인지 아닌지 파악해서 2개면 레이블인코딩,아니면 원핫인코딩하기
                    if data.nunique() <= 2:
                        # 레이블인코딩
                        label_encoder = LabelEncoder()
                        X_new[name] = label_encoder.fit_transform(data)
                        
                    
                    else :
                        #원핫인코딩
                        ct = ColumnTransformer( [ ('encoder',OneHotEncoder(), [0])  ] , remainder='passthrough' )
                        
                        col_names = sorted(data.unique())
                        
                        X_new[  col_names  ] = ct.fit_transform(  data.to_frame()  )
                        
                else :
                    # 숫자 데이터 처리
                    X_new[name] = data
            scaler = MinMaxScaler()
            X_new = scaler.fit_transform(X_new)
            st.dataframe(X_new)

            st.subheader('wcss를 위한 클러스터링 갯수를 선택')

            #행의 갯수 가저오기
            if X_new.shape[0] < 10:
                default_value = X_new.shape[0]
            else :
                default_value = 10

            max_number = st.slider('최대 그룹 선택',2,20,value= default_value)


            wcss = []
            for k in np.arange(1, max_number+1) :
                kmeans = KMeans(n_clusters= k, random_state=5)
                kmeans.fit(X_new)
                wcss.append( kmeans.inertia_ )

            ##5.엘보우 메소드 차트를 화면에 표시
            fig1 = plt.figure()
            x = np.arange(1, max_number +1)
            plt.plot( x, wcss )
            plt.title('The Elbow Method')
            plt.xlabel('Number of Clusters')
            plt.ylabel('WCSS')
            st.pyplot(fig1)     


            ## 6. 실제 그룹핑하고 싶은 갯수를 입력
            #k = st.slider('그룹 갯수 결정',1,max_number)

            k =st.number_input('그룹 갯수 결정',1,max_number)
            kmeans = KMeans(n_clusters= k, random_state=5)

            y_pred = kmeans.fit_predict(X_new)

            df['Group'] = y_pred

            ##7.위에서 입력한 그룹의 갯수로 클러스터링하여 결과를 보여준다
            st.dataframe(df.sort_values('Group'))

            df.to_csv('result.csv')

## 1.비어있는 데이터 처리가 안되어있다
#2.숫자로 되어있는 컬럼만 했는데 문자로 되어있는 컬럼 처리
  #카테고리컬 데이터가 2개냐 3개이상이냐(3개이상이면 원핫)
#3. 유저 인터랙티브한 데이터 분석 기능 추가

if __name__ == '__main__' :
    main()








