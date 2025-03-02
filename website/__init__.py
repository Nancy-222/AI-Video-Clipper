from flask import Flask

def create_app():
    app=Flask(__name__) #initialize the app
    app.config['SECRET_KEY']='AIVideoClipper' #encrypt and secure cookies and session data it can be whatever

    from .views import views
    app.register_blueprint(views, url_prefix='/')
    return app