FROM ubuntu:20.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# C++ dependencies
RUN apt-get update && \
    apt-get install -y build-essential cmake wget git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda with Python 3.8+
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN wget -P /tmp/ https://repo.anaconda.com/miniconda/Miniconda3-py38_23.5.2-0-Linux-x86_64.sh && \
    /bin/bash /tmp/Miniconda3-py38_23.5.2-0-Linux-x86_64.sh -b -p /opt/conda && \
    rm /tmp/Miniconda3-py38_23.5.2-0-Linux-x86_64.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    /opt/conda/bin/conda clean -afy

# Install cityflow 
RUN git clone https://github.com/cityflow-project/CityFlow.git /home/CityFlow 
RUN
RUN pip install --no-cache-dir flask && \
    cd /home/CityFlow && \
    pip install --no-cache-dir .
RUN pip3 install torch  
COPY requirements.txt /home
#RUN pip install -r /home/requirements.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -r /home/requirements.txt  
RUN pip install tensorflow
RUN apt-get update
RUN python3 -m pip install --upgrade typing_extensions
RUN python -m pip install rich
WORKDIR /home/code
