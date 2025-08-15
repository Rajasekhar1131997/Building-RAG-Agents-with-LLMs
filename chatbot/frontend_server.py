
## NOTE: THIS SERVER IS RUNNING PERPETUALLY FOR THIS COURSE.
## DO NOT CHANGE CODE HERE; INSTEAD, INTERFACE WITH IT VIA USER INTERFACE
## AND BY DEPLOYING ON PORT :9012

import gradio as gr
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

route = os.environ.get("APP_ROOT_PATH")

#####################################################################
## Final App Deployment

from frontend_block import get_demo

demo = get_demo()
demo.queue()

logger.warning("Starting FastAPI app")
app = FastAPI()

app.mount("/imgs", StaticFiles(directory="/app/imgs"), name="images")
app.mount("/slides", StaticFiles(directory="/app/slides"), name="slides")
gr.set_static_paths(paths=["imgs", "slides"])
app = gr.mount_gradio_app(app, demo, '/', root_path=route)

@app.route("/health")
async def health():
    return {"success": True}, 200
