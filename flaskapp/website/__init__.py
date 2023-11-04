from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from os import path
import os
from flask_login import LoginManager

db = SQLAlchemy()
DB_NAME = "database.db"


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'hjshjhdjah kjshkjdhjs'
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    APP_ROOT = path.dirname(path.abspath(__file__))

    # create folder to save new employee image
    UPLOAD_FOLD = 'my_db'
    UPLOAD_FOLDER = path.join(APP_ROOT, UPLOAD_FOLD)
    isExist = path.exists(UPLOAD_FOLDER)
    
    if not isExist:

        # Create a new directory because it does not exist
        os.makedirs(UPLOAD_FOLDER)
        app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
        print("The new directory is created!")
    else:
        app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    db.init_app(app)

    from .views import views
    from .auth import auth

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')

    from .models import User #, Note
    
    with app.app_context():
        db.create_all()

    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))

    return app


def create_database(app):
     if not path.exists('website/' + DB_NAME):
         db.create_all(app=app)
         print('Created Database!')
