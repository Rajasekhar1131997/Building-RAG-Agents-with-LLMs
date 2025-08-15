# Use a base image with Python
FROM python:3.11-slim

# Set working directory
WORKDIR /dli

RUN apt-get update && apt-get install -y apt-utils
RUN apt-get install -y build-essential bash curl unzip wget git libgl1

#Run pip dependencies
COPY composer/requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install ipywidgets jupyterlab==4.0.0 jupyter-archive==3.4.0

# COPY notebooks
WORKDIR /dli/task
# COPY . .

# Disable the startup privacy/news prompt (Announcements)
RUN jupyter labextension disable "@jupyterlab/apputils-extension:announcements"

# Expose port 8888 for JupyterLab
EXPOSE 8888 
EXPOSE 9012

# Start JupyterLab when the container runs
ADD composer/entrypoint.sh /usr/local/bin
RUN chmod +x /usr/local/bin/*.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]