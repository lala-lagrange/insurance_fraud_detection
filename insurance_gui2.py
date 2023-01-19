#IMPORTING ALL THE NECESSARY LIBRARIES
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import matplotlib.pyplot as plt


#DATA PREPROCESSING
le = LabelEncoder()
df = pd.read_csv('/home/ak/Desktop/ML/DATASETS/INFIDATA/SAMPLE/insurance_claims_mod.csv')
df_org = pd.read_csv('/home/ak/Desktop/ML/DATASETS/INFIDATA/SAMPLE/insurance_claims_mod.csv')

df.replace('?','UNKNOWN')

df['insured_sex'] = le.fit_transform(df['insured_sex'])
df['insured_occupation'] = le.fit_transform(df['insured_occupation'])
df['insured_relationship'] = le.fit_transform(df['insured_relationship'])
df['incident_severity'] = le.fit_transform(df['incident_severity'])
df['property_damage'] = le.fit_transform(df['property_damage'])
df['police_report_available'] = le.fit_transform(df['police_report_available'])
df['collision_type'] = le.fit_transform(df['collision_type'])
df['insured_education_level'] = le.fit_transform(df['insured_education_level'])
df['fraud_reported'] = le.fit_transform(df['fraud_reported'])
df['policy_state'] = le.fit_transform(df['policy_state'])
df['auto_make'] = le.fit_transform(df['auto_make'])
df['auto_year'] = le.fit_transform(df['auto_year'])

x = df.iloc[:,:-1]
y = df.iloc[:,-1]
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.4,random_state=0)

sc  = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#creating seaborn environment
import seaborn as sns
sns.color_palette("Spectral", as_cmap=True)
plt.style.use('fivethirtyeight')
ax = sns.countplot(x='fraud_reported', data=df_org, hue='fraud_reported')

#creating a streamlit page

st.title("INSURANCE FRAUD PREDICTOR")
st.sidebar.title("INPUT PARAMETERS")

age = st.sidebar.number_input("ENTER AGE : ",step=1,min_value=5)
insured_sex1={'MALE':'1','FEMALE':'0'}
insured_sex = st.sidebar.selectbox("ENTER SEX",options={'MALE','FEMALE'})
insured_sex=insured_sex1[insured_sex]

insured_occupation1 = {'craft-repair':'2', 'machine-op-inspct':'6', 'sales':'11', 'armed-forces':'1',
       'tech-support':'12', 'prof-specialty':'9', 'other-service':'7',
       'priv-house-serv':'8', 'exec-managerial':'3', 'protective-serv':'10',
       'transport-moving':'13', 'handlers-cleaners':'5', 'adm-clerical':'0',
       'farming-fishing':'4'}
insured_occupation = st.sidebar.selectbox("ENTER OCCUPATION : ",options={'craft-repair':'2', 'machine-op-inspct':'6', 'sales':'11', 'armed-forces':'1',
       'tech-support':'12', 'prof-specialty':'9', 'other-service':'7',
       'priv-house-serv':'8', 'exec-managerial':'3', 'protective-serv':'10',
       'transport-moving':'13', 'handlers-cleaners':'5', 'adm-clerical':'0',
       'farming-fishing':'4'})
insured_occupation = insured_occupation1[insured_occupation]

insured_relationship1 = {'husband':'0', 'other-relative':'2', 'own-child':'3', 'unmarried':'4', 'wife':'5',
       'not-in-family':'1'}
insured_relationship = st.sidebar.selectbox("ENTER INSURED RELATIONSHIP : ",options={'husband':'0', 'other-relative':'2', 'own-child':'3', 'unmarried':'4', 'wife':'5',
       'not-in-family':'1'})
insured_relationship = insured_relationship1[insured_relationship]

incident_severity1 = {'Major Damage':'0', 'Minor Damage':'1', 'Total Loss':'2', 'Trivial Damage':'3'}
incident_severity = st.sidebar.selectbox("ENTER INSURED SEVERITY : ",options={'Major Damage':'0', 'Minor Damage':'1', 'Total Loss':'2', 'Trivial Damage':'3'})
incident_severity = incident_severity1[incident_severity]

property_damage1 = {'YES':'2', 'UNKNOWN':'0', 'NO':'1'}
property_damage = st.sidebar.selectbox("ENTER PROPERTY DAMAGE : ",options={'YES':'2', 'UNKNOWN':'0', 'NO':'1'})
property_damage = property_damage1[property_damage]

police_report_available1 = {'YES':'2', 'UNKNOWN':'0', 'NO':'1'}
police_report_available = st.sidebar.selectbox("IS POLICE REPORT AVAILABLE : ",options={'YES':'2', 'UNKNOWN':'0', 'NO':'1'})
police_report_available = police_report_available1[police_report_available]

collision_type1 = {'Side Collision':'3', 'UNKNOWN':'0', 'Rear Collision':'2', 'Front Collision':'1'}
collision_type = st.sidebar.selectbox("ENTER COLLISION TYPE : ",options={'Side Collision':'3', 'UNKNOWN':'0', 'Rear Collision':'2', 'Front Collision':'1'})
collision_type = collision_type1[collision_type]

policy_state1 = {'OH':'2', 'IN':'1', 'IL':'0'}
policy_state = st.sidebar.selectbox("ENTER POLICY STATE : ",options={'OH':'2', 'IN':'1', 'IL':'0'})
policy_state = policy_state1[policy_state]

insured_education_level1 = {'MD':'4', 'PhD':'6', 'Associate':'0', 'Masters':'5', 'High School':'2', 'College':'1',
       'JD':'3'}
insured_education_level = st.sidebar.selectbox("ENTER INSURED EDUCATION LEVEL : ",options={'MD':'4', 'PhD':'6', 'Associate':'0', 'Masters':'5', 'High School':'2', 'College':'1',
       'JD':'3'})
insured_education_level = insured_education_level1[insured_education_level]


auto_make1 = {'Saab':'10', 'Mercedes':'8', 'Dodge':'4', 'Chevrolet':'3', 'Accura':'0', 'Nissan':'9',
       'Audi':'1', 'Toyota':'12', 'Ford':'5', 'Suburu':'11', 'BMW':'2', 'Jeep':'7', 'Honda':'6',
       'Volkswagen':'13'}
auto_make = st.sidebar.selectbox("ENTER AUTO MAKE : ",options={'Saab':'10', 'Mercedes':'8', 'Dodge':'4', 'Chevrolet':'3', 'Accura':'0', 'Nissan':'9',
       'Audi':'1', 'Toyota':'12', 'Ford':'5', 'Suburu':'11', 'BMW':'2', 'Jeep':'7', 'Honda':'6',
       'Volkswagen':'13'})
auto_make = auto_make1[auto_make]

auto_year = st.sidebar.number_input('ENTER MANUFACTURED YEAR : ',min_value=1995,max_value=2023,step=1)

#TRAINING THE MODEL
model = LogisticRegression(random_state=0)
model.fit(X_train,Y_train)

new = [[age,insured_sex,insured_occupation,insured_relationship,incident_severity,property_damage,police_report_available,collision_type,policy_state,insured_education_level,auto_make,auto_year]]
result = model.predict(sc.transform(new))



#PRINTING MODEL PARAMETERS
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import classification_report
y_pred = model.predict(X_test)
cm = confusion_matrix(Y_test,y_pred)

col1, col2, col3 = st.columns(3)
with col1:
    pass
with col2:
    submit = st.sidebar.button("SUBMIT")
with col3 :
    pass

if submit:
    if result == 0:
      st.success('NO POSSIBILITY OF FRAUD DETECTED')
      st.subheader('CLASSIFICATION REPORT')

    else:
      st.error('THERE IS A POTENTIAL FRAUD')

    tab1, tab2,tab3 = st.tabs(['MODEL PARAMETERS', 'CLASSIFICATION REPORT', 'MODEL GRAPHS'])
    with tab1:
        st.header('MODEL PARAMETERS')
        st.info("ACCURACY : {0}%".format(accuracy_score(Y_test, y_pred) * 100))
        st.info("PRECISION : {0}%".format(precision_score(Y_test, y_pred) * 100))
        st.info("RECALL : {0}%".format(recall_score(Y_test, y_pred) * 100))
        st.info("F1 SCORE : {0}%".format(f1_score(Y_test, y_pred) * 100))
    with tab2:
        st.header('CLASSIFICATION REPORT')
        report = classification_report(Y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, height=212, width=1000)
    with tab3:
        st.header('MODEL GRAPHS')
        plt.style.use('fivethirtyeight')
        fig = plt.figure(figsize=(10, 6))
        ax = df_org.groupby('insured_sex').fraud_reported.count().plot.bar(ylim=0)
        ax.set_ylabel('Fraud reported')
        st.pyplot(fig=fig)

        plt.style.use('fivethirtyeight')
        fig = plt.figure(figsize=(10, 6))
        ax = df_org.groupby('insured_occupation').fraud_reported.count().plot.pie(ylim=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
        ax.set_ylabel('Fraud reported')
        st.pyplot(fig=fig)

        plt.style.use('fivethirtyeight')
        fig = plt.figure(figsize=(10, 6))
        ax = df_org.groupby('incident_severity').fraud_reported.count().plot.bar(ylim=0)
        ax.set_ylabel('Fraud reported')
        st.pyplot(fig=fig)

        plt.style.use('fivethirtyeight')
        fig = plt.figure(figsize=(10, 6))
        ax = df_org.groupby('police_report_available').fraud_reported.count().plot.bar(ylim=0)
        ax.set_ylabel('Fraud reported')
        st.pyplot(fig=fig)

        plt.style.use('fivethirtyeight')
        fig = plt.figure(figsize=(10, 6))
        ax = df_org.groupby('collision_type').fraud_reported.count().plot.bar(ylim=0)
        ax.set_ylabel('Fraud reported')
        st.pyplot(fig=fig)

        plt.style.use('fivethirtyeight')
        fig = plt.figure(figsize=(10, 6))
        ax = df.groupby('collision_type').fraud_reported.count().plot.bar(ylim=0)
        ax.set_ylabel('Fraud reported')
        st.pyplot(fig=fig)




else:
    st.error('COMPLETE ALL PARAMETERS')




