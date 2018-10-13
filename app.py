from flask import Flask,render_template,request,json
from NeuralNet_l import func

input_values={}

app=Flask(__name__)

@app.route('/',methods=['POST','GET'])
def home():
	pred=[[0]]
	if request.method=='POST':
		xt=[]
		input_values['Age']=request.form['Age']
		input_values['TB']=request.form['TB']
		input_values['DB']=request.form['DB']
		input_values['AAP']=request.form['AAP']
		input_values['SGPT']=request.form['SGPT']
		input_values['SGOT']=request.form['SGOT']
		input_values['TP']=request.form['TP']
		input_values['ALB']=request.form['ALB']
		input_values['AGR']=request.form['AGR']
		input_values['Gender']=request.form['Gender']
		attr=['Age','TB','DB','AAP','SGPT','SGOT','TP','ALB','AGR','Gender']
		for i in attr:
			xt.append(float(input_values[i]))
		pred=func(xt)
		pred=pred*100
		print(pred)

	return render_template('home.html',pred=pred)


if __name__ == '__main__':
	app.run(debug=True)