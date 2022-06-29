import numpy as np
import pandas as pd
import csv
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import json

# Flask constructor
import flask
from DBConnection import Db
from flask import request
from flask import Flask, render_template,redirect

from DBConnection import Db

app = Flask(__name__, template_folder='template')

u_name = ""


def eg(res):

    global u_name

    age = res.get('age')
    c_w = res.get('c_w')
    y_c = res.get('y_c')
    y_curr = res.get('y_curr')
    js = res.get('js')
    es = res.get('es')
    wb = res.get('wb')
    ji = res.get('ji')
    l_pro = res.get('l_pro')
    c_mng = res.get('c_mng')
    wh = res.get('wh')
    value1 = res.get('value1')
    value2 = res.get('value2')
    value3 = res.get('value3')
    mi = res.get('mi')
    gen = res.get('gen')
    ms = res.get('ms')
    bt = res.get('bt')
    ot = res.get('ot')
    t_w_y = res.get('t_w_y')
    t_p = res.get('t_p')
    dr = mi/20
    hr = dr/wh
    xyz=''

    df = pd.read_csv(r"C:\Users\Jeevan Jose\Desktop\WA_Fn-UseC_-HR-Employee-Attrition.csv")
    # print(df.head(7))
    df = df.drop('Over18', axis=1)
    df = df.drop('EmployeeNumber', axis=1)
    df = df.drop('StandardHours', axis=1)
    df = df.drop('EmployeeCount', axis=1)
    df = df.drop('Education', axis=1)
    df = df.drop('JobLevel', axis=1)
    df = df.drop('MonthlyRate', axis=1)
    df = df.drop('PercentSalaryHike', axis=1)
    df = df.drop('PerformanceRating', axis=1)
    df = df.drop('RelationshipSatisfaction', axis=1)
    df = df.drop('StockOptionLevel', axis=1)
    df = df.drop('DistanceFromHome', axis=1)

    for column in df.columns:
        if df[column].dtype == np.float64:
            continue
        df[column] = LabelEncoder().fit_transform(df[column])

    # print(df.head(10))
    df['Age_Years'] = df['Age']
    df = df.drop('Age', axis=1)

    x = df.iloc[:, 1:df.shape[1]].values
    y = df.iloc[:, 0].values

    # Split the dataset into 75% training and 25% testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    # Use the random forest classifier module
    forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    forest.fit(x_train, y_train)

    # Get the accuracy on the training dataset
    print(forest.score(x_train, y_train))

    # Show the confusion matrix and accuracy score for the model on the test data
    cm = confusion_matrix(y_test, forest.predict(x_test))

    TN = cm[0][0]
    TP = cm[1][1]
    FN = cm[1][0]
    FP = cm[0][1]

    print(cm)
    print('Random forestModel Testing Accuracy -> {}'.format((TP + TN) / (TP + TN + FN + FP)))

    categ = ['Employee will stay', 'Employee will leave']
    custom_dt = [[bt, dr, value1, value2,es,gen,hr,ji,value3,js, ms,mi,c_w,ot,t_w_y,t_p,wb, y_c, y_curr, l_pro, c_mng,age]]
    print(categ[int(forest.predict(custom_dt))])



    att = categ[int(forest.predict(custom_dt))]

    db = Db()
    sv = db.selectOne("select * from attrition where email='" + u_name + "'")
    if sv is None:
        tmp = db.selectOne("select * from login where email='" + u_name + "'")
        db.insert("insert into attrition VALUES('" + str(tmp['id']) + "','"+u_name+"','"+att+"')")
        return '''<script> alert('Submitted')</script>'''
    else:
        print("You have already attended the questionnaire")
        return '''<script> alert('You have already attended the questionnaire')</script>'''


@app.route('/')
def handle_data():
    return render_template('ind.html')


@app.route('/first_page')
def first_page():
    global u_name
    u_name = ""
    return render_template('ind.html')


@app.route('/adm_reg', methods=['get', 'post'])
def adm_reg():
    if request.method == "POST":
        name = request.form['name']
        email = request.form['email']
        passw = request.form['password']
        typ = "admin"
        db = Db()
        db.insert("insert into login VALUES('','"+email+"','"+passw+"')")
        db.insert("insert into registration VALUES('','" + name + "','" + email + "','" + passw + "','" + typ + "')")
        return '''<script> alert('User Registered');window.location="/emp_log"</script>'''
    else:
        return render_template('hr_reg.html')


@app.route('/emp_reg', methods=['get', 'post'])
def emp_reg():
    if request.method == "POST":
        name = request.form['name']
        phone = str(request.form['phone'])
        email = request.form['email']
        passw = request.form['password']
        gender = request.form['fav_language']
        date = request.form['birthday']
        city = request.form['City']
        pin = str(request.form['pin'])
        state = request.form['state']
        quali = request.form['qual']
        prof = request.form['prof']
        typ = "employee"
        status = "pending"
        db = Db()
        db.insert("insert into register VALUES('','" + name + "','" + phone + "','" + gender + "','" + date + "','" + city + "','" + pin + "','" + state + "','" + quali + "','" + prof + "','" + email + "','" + passw + "')")
        temp = db.selectOne("select * from register where email='" + email + "'")
        xy = temp['id']
        db.insert("insert into login VALUES('" + str(xy) + "','" + email + "','" + passw + "','" + typ + "','" + status + "')")
        return '''<script> alert('User Registered');window.location="/"</script>'''

    else:
        return render_template('cust_reg.html')


@app.route('/adm_log', methods=['get', 'post'])
def adm_log():
    global u_name
    if request.method == "POST":
        em = request.form['username']
        pa = request.form['pass']
        db = Db()
        ss = db.select("select * from login where user_name='" + em + "' and pass='" + pa + "'")
        if ss is not None:
            u_name = str(em)
            return redirect('/adminhome')
        else:
            return '''<script> alert('User Not Found');window.location="/"</script>'''
    else:
        return render_template('hr_log.html')


@app.route('/emp_log', methods=['get', 'post'])
def emp_log():
    global u_name
    if request.method == "POST":
        em = request.form['username']
        pa = request.form['pass']
        db = Db()
        ss = db.selectOne("select * from login where email='" + em + "' and password='" + pa + "'")
        if ss is not None and ss['status'] == 'user':
            u_name = str(em)
            if ss['typ'] == 'admin':
                return redirect('/adminhome')
            else:
                return redirect('/emphome')
        elif ss is not None and ss['status'] == 'pending':
            return '''<script> alert('No permission yet');window.location="/"</script>'''
        else:
            return '''<script> alert('User Not Found');window.location="/"</script>'''
    else:
        return render_template('cust_log.html')


@app.route('/adminhome')
def adminhome():
    return render_template('dashboard_adm.html', data=u_name)


@app.route('/emphome')
def emphome():
    return render_template('dashboard_emp.html', data=u_name)


@app.route('/test', methods=['POST'])
def test():
    output = request.get_json()
    result = json.loads(output)
    print(result)
    eg(result)
    return result


@app.route('/view_users')
def view_users():
    db = Db()
    res = db.select("select * from register")
    return render_template('users.html', data=res)


@app.route('/emp_details')
def emp_details():
    db = Db()
    res = db.select("select * from attrition")
    return render_template('attrition_or_not.html', data=res)


@app.route('/profile')
def profile():
    db = Db()
    res = db.select("select * from register where email='" + u_name + "'")
    return render_template('user.html', data=res)


@app.route('/questionnaire')
def questionnaire():
    db = Db()
    t1 = db.selectOne("select * from attrition where email='" + u_name + "'")
    if t1 is not None:
        return render_template('q_fin.html', data=u_name)
    else:
        return render_template('questionnaire.html', data=u_name)


@app.route('/employees')
def employees():
    db = Db()
    res = db.select("select * from register")
    return render_template('employees.html', data=res)


@app.route('/quest_res')
def quest_res():
    db = Db()
    res = db.select("select * from attrition,register where attrition.email=register.email")
    return render_template('quest_result.html', data=res)


@app.route('/approve')
def approve():
    db = Db()
    res = db.select("select * from login,register where login.status='pending' and login.email=register.email")
    return render_template('approve_user.html', data=res)


@app.route('/approveuser/<email>')
def approveuser(email):
    db = Db()
    res = db.update("update login set status='user' where email='" + str(email) + "'")
    return '''<script> alert('User Approved');window.location="/approve"</script>'''


@app.route('/disapproveuser/<email>')
def disapproveuser(email):
    db = Db()
    db.delete("delete from login where email='" + str(email) + "'")
    db.delete("delete from register where email='" + str(email) + "'")
    return '''<script> alert('User Disapproved');window.location="/approve"</script>'''


@app.route('/notifications', methods=['POST', 'GET'])
def notifications():
    db = Db()
    if request.method == "POST":
        em = request.form['g_id']
        mess = request.form['ta']
        db.insert("insert into notify values('" + em + "','" + mess + "')")
    return render_template('notifications.html')


@app.route('/notifications2', methods=['POST', 'GET'])
def notifications2():
    db = Db()
    res = db.select("select * from notify where gmail_id='" + u_name + "'")
    return render_template('notifications2.html', data=res)


@app.route('/fp')
def fp():
    return render_template('for_pass.html')


@app.route('/res_pass', methods=["POST", "GET"])
def res_pass():
        db = Db()
        em = request.form.get("m_id")
        mess = request.form.get("pass_wd")
        t1 = db.update("update register set password='" + mess + "' where email='" + str(em) + "'")
        t2 = db.update("update login set password='" + mess + "' where email='" + str(em) + "'")
        return redirect('\emp_log')


if __name__ == '__main__':
    app.run()


