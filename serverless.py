from functions.app import create_app

def handler(event, context):
    app = create_app()
    return app(event, context)
