# RNN

> 순환 신경망(Recurrent Neural Network, RNN)

## 뉴런과 층을 셀Cell이라고 한다.

## 출력은 은닉 상태Hidden state라고 한다.

- RNN은 Sequence 모델. 입력과 출력을 시퀀스 단위로 처리하는 모델.
- 출력에 해당되는 번역된 문장 또한 단어 시퀀스.
- 이러한 시퀀스들을 처리하기 위해 고안된 모델들을 시퀀스 모델이라 한다. 
- RNN은 가장 기본적인 시퀀스 모델
- Xt -> Cell -> Yt 
- 입력 벡터 -> 셀 -> 출력벡터
- 셀: 이전의 값을 기억하려고 하는 일종의 메모리 역할을 하는 노드. == 메모리 셀, RNN 셀 이라고도 함.
- 은닉층의 메모리 셀은 각각의 시점에서 바로 이전 시점의 은닉층의 메모리 셀에서 나온 값을 자신의 입력으로 사용하는 재귀적 활동을 하고 있다.
- 변수 t: 현재 시점
- 현재 시점 t에서의 메모리 셀이 갖고 있는 값은 과거의 메모리 셀들의 값에 영향을 받은 것임을 의미.
- 메모리 셀이 출력층 방향으로 또는 다음 시점 t+1의 자신에게 보내는 값을 은닉 상태Hidden state라고 한다.
- t 시점의 메모리 셀: t-1 시점의 메모리 셀이 보낸 은닉 상태 값을 t 시점의 은닉 상태 계산을 위한 입력값으로 사용

---

- ## RNN

  - 일 대 다, 다 대 일, 다 대 다

  - 입력, 출력의 길이를 다양한 용도로 사용 가능

  - 입력과 출력의 길이에 따라서 달라지는 RNN 형태를 보여둔다.

  - RNN 셀의 각 시점별 입, 출력의 단위는 사용자가 정의하기 나름이지만, 가장 보편적인 단어는 '단어 벡터'이다.

  -  ##  '일 대 다' 출력(one-to-many) 모델

  - ![1](https://user-images.githubusercontent.com/57430754/77276435-00d9fb00-6cfe-11ea-92ca-c91a99fefef9.png)

  - 하나의 이미지 입력에 대해 사진의 제목을 출력하는 '이미지 캡셔닝Image Captioning' 작업에 활용. 사진의 제목은 단어들의 나열이므로 '시퀀스 출력'

  - ## '다 대 일' 출력(many-to-one) 모델

  - ![2](https://user-images.githubusercontent.com/57430754/77276438-01729180-6cfe-11ea-8e82-784963c2e02f.png)

  - 입력 문서가 긍정적인지 부정적인지 판별하는 감성 분류Sentiment classification, 또는 메일이 정상인지, 스팸인지 판별하는 스팸 메일 분류Spam detection

  - ## '다 대 다' 출력(many-to-many) 모델

  - ![3](https://user-images.githubusercontent.com/57430754/77276440-020b2800-6cfe-11ea-877e-dd8f829fa1f1.png)

  - 입력 문장으로부터 대답 문장을 출력하는 챗봇, 입력 문장으로부터 번역된 문장을 출력하는 번역기, 개체명 인식이나 품사 태깅과 같은 작업이 속함.

  - 개체명 인식을 수행할 때의 RNN 아키텍처를 보여줌.

  - 츨력층의 결과값인 yt를 계산하기 위한 활성화 함수로는 이진 분류: 시그모이드 함수, 다양한 카테고리: 소프트 맥스 함수

  ---

  - Python Numpy로 RNN 구현

  - ```python
    hidden_state_t = 0 # 초기 은닉 상태를 0(벡터)로 초기화
    for input_t in input_length: # 각 시점마다 입력을 받는다.
        output_t = tanh(input_t, hidden_state_t)  # 각 시점에 대해 입력과 은닉 상태를 가지고 연산
        hiden_state_t = output_t  # 계산 결과는 현재 시점의 은닉 상태가 된다.
    ```

  - hidden_state_t: t 시점의 은닉 상태

  - input_length: 입력 데이터의 길이

  - timesteps: 총 시점의 수 == 입력 데이터의 길이

  - input_t: t 시점의 입력값

  - 파이토치의 경우 (batch_size, timesteps, input_size)의 크기의 3D 텐서를 입력으로 받는 것을 기억

  - ```python
    import numpy as np
    
    temesteps = 10  # 시점의 수, NLP에서는 보통 문장의 길이.
    input_size = 4  # 입력의 차원. NLP에서는 보통 단어 벡터의 차원.
    hidden_size = 8  # 은닉 상태의 크기. 메모리 셀의 용량.
    
    inputs = np.randam.random.((timesteps, input_size))  # 입력에 해당하는 2D 텐서
    
    hidden_state_t = np.zeros((hidden_size,))  # 초기 은닉 상태는 0(벡터)으로 초기화
    # 은닉 상태의 크기 hidden_size로 은닉 상태를 만듦.
    ```

  - 우선 시점, 입력 차원, 은닉 상태 크기 설정 및 초기 은닉 상태 정의

  - 현재 초기 은닉 상태는 0의 값을 가지는 벡터로 초기화

  - ```python
    # 초기 은닉 상태 출력
    
    print(hidden_state_t)  # 8의 크기를 가지는 은닉 상태. 현재는 초기 은닉 상태로 모든 차원이 0의 값을 가짐.
    ```

  - 은닉 상태 크기를 8로 정의했으므로 8의 차원을 가지는 0의 값으로 구성된 벡터 출력.

  - ```python
    # 가중치, 편향 정의
    
    Wx = np.random.random((hidden_size, input_size))  # (8, 4) 크기의 2D 텐서 생성. 입력에 대한 가중치.
    Wh = np.random.random((hidden_size, input_size))  # (8, 8) 크기의 2D 텐서 생성. 입력에 대한 가중치.
    b = np.random.random((hidden_size,))  # (8,) 크기의 1D 텐서 생성. 이 값은 편향(bias).
    ```

  - ```python
    # 가중치, 편향 크기 출력
    
    print(np.shape(Wx))
    print(np.shape(Wh))
    print(np.shape(b))
    ```

  - Wx = 은닉 상태의 크기 X 입력의 차원

  - Wh = 은닉 상태의 크기 X 은닉 상태의 크기

  - b = 은닉 상태의 크기

  - ```python
    # 모든 시점의 은닉 상태를 출력한다고 가정하고, RNN 층을 동작
    
    total_hidden_states = []
    
    # 메모리 셀 동작
    for input_t in inputs:  # 각 시점에 따라 입력값이 입력됨
        output_t = np.tanh(np.dot(Wx, input_t) + np.dot(Wh, hidden_state_t) + b)  # Wx * Xt * Wh * Ht-1 + b(bias)
        total_hidden_states.append(list(output_t))  # 각 시점의 은닉 상태의 값을 계속 축적
        print(np.shape(total_hidden_states)) # 각 시점 t별 메모리 셀의 출력의 크기는 (timestep, output_dim)
        hidden_state_t = output_t
        
    total_hidden_states = np.stack(total_hidden_state, axis = 0)
    # 출력 시 값을 깔끔하게 해준다.
    
    print(total_hidden_state)  # (timesteps, output_dim)의 크기. 이 경우 (10, 8)의 크기를 가지는 메모리 셀의 2D 텐서를 출력.
    ```

    









