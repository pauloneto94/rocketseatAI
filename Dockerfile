FROM public.ecr.aws/lambda/python:3.12

RUN microdnf update -y && microdnf install -y gcc-c++ make

COPY requirements.txt ${LAMBDA_TASK_ROOT}

RUN pip install -r requirements.txt

COPY travel_agent.py ${LAMBDA_TASK_ROOT}

RUN chmod +x travel_agent.py

CMD ["travel_agent.lambda_handler"]