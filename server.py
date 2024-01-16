from flask import Flask, request, jsonify
from chatbot_model import ChatbotModel
from flask_cors import CORS, cross_origin



app = Flask(__name__)
cors = CORS(app)

chatbotModel = ChatbotModel()

chatbotModel.loadData()
chatbotModel.train()


@app.route('/chatbot', methods=['POST'])
@cross_origin()
def load_text():
    question = request.form.get('question')
    context = request.form.get('context')
    return chatbotModel.loadText(question, context)

@app.route('/question', methods=['POST'])
def question_answer():
    question = request.form.get('question')
    return chatbotModel.chatbot(question)

@app.route('/hello', methods=['GET'])
def hello():
    return 'OK'


app.run()