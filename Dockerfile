FROM pytorch/pytorch:latest

RUN apt update -y && apt install git -y
RUN pip install stable-baselines3 shimmy tensorflow tensorboard
RUN git clone https://github.com/cage-challenge/cage-challenge-2.git

RUN sed -i 's/operating_system\[hostname\]/operating_system.get(hostname)/' cage-challenge-2/CybORG/CybORG/Shared/Actions/AbstractActions/PrivilegeEscalate.py
#RUN sed -i 's/info2\] == hostname/info.get(2) == hostname/' cage-challenge-2/CybORG/CybORG/Agents/Wrappers/RedTableWrapper.py


RUN pip install -e ./cage-challenge-2/CybORG/

COPY . .

ENV TF_ENABLE_ONEDNN_OPTS 0

ENTRYPOINT ["/bin/bash", "-c"]