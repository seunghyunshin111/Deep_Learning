# Tensorflow + Anaconda + Jupyter + Keras

- py36 가상환경에 keras 설치 및 텐서플로우1 설치
- pip install jupyter
- pip install  numpy pandas
- pip install matplotlib
- pip install keras
- low level 수준의 유지, 보수 엔지니어가 아닌 이상 텐서플로우1 대신, high level 수준인 텐서플로우2를 쓴다.

<br>

- ## Tensorflow: 프레임워크(다양한 종류가 있다.)

  - 대규모 머신러닝에 맞춰 프로그래밍이 가능하지만 디버깅이 많고 무겁다.

  - 데이터의 패턴을 찾는 것 (대수학 등의 연산을)

  - 데이터를 넣으면 플로우를 연산을 거쳐 또 다른 연산을 뽑아내는 것이 텐서플로우

  - 텐서플로의 그래프는 사이클이 있어도 되고, Node와 Edge로 구성된 비선형 구조

  - Node: 연산

  -  텐서(Tensorflow의 데이터) + 연산

  - 텐서플로는 노드에 연산 operator, 변수 variable, 상수 Constant등을 정의하고, 노드 간의 연결인 엣지를 통해 텐서를 주고받으면서 계산을 수행한다.

  - ### 텐서플로우에서 행렬의 곱셈은 일반 * 를 사용하지 않고, 텐서플로우 함수 “tf.matmul” 을 사용한다.

  - None: 행을 flexible하게 받겠다는 의미

<br>

- java는 완벽한 객체 지향이라 프로그램이 무거워질 수밖에 없다.

- ## Pytorch: 텐서플로처럼 프레임워크 중 하나