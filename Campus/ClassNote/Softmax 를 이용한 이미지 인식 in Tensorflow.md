# Softmax 를 이용한 이미지 인식

- 55000: 데이터

- 784: 이미지 하나가 가지고 있는 숫자 (속성)

- 10: 최종적으로 10개의 숫자를 감별하는 결과가 나와야 한다. W값 = 10, 10개의 값은 각각 784개의 숫자에 적용해야 한다.

  b는 10개의 값에 각각 더하는 값이기 때문에, 크기가 10인 행렬이 된다.

  ![Softmax](https://user-images.githubusercontent.com/57430754/76477962-55ea5700-644a-11ea-93a4-0831f746b4f2.png)

- ```python
  # Softmax regression
  # Tensorflow's code
  
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  k = tf.matmul(x, W) + b
  y = tf.nn.softmax(k)
  ```

- 브로드 캐스팅: 차원이 다른 행렬을 큰 행렬의 크기로 늘려주는 기능

- b: 원래 1x10 -> 55000x10

- 코스트(비용) 함수: 크로스 엔트로피 함수의 평균을 이용

- ```python
  Cost = tf.reduce_mean(
  tf.nn.softmax_cross_entropy_with_logits(tf.matmul(x, W) + b, y_))
  ```

- 가설에 의해 계산된 값 y 대신 tf.matmul(x,W)+b를 넣은 이유는 tf.nn.softmax_cross_entropy_with_logits 함수 자체가 softmax를 포함하기 때문

- y_는 학습을 위해 입력된 값

<br>

- 층이 적더라도 노드 수가 많으면, 다양하게 본다.
- 은닉층(중간층)은 절대 시그모이드/소프트맥스를 쓰지 않고 ReLu를 쓴다.
- 출력층(마지막층)은 반드시 시그모이드, 소프트맥스를 쓴다.
- 층이 깊어질수록 시그모이드는 안 되기 때문에 ReLu를 쓴다.

<br>

## 텐서플로우로 구현한 전체 코드 해석!

- ```python
  # Import data
  from tensorflow.examples.tutorials.mnistimport input_data
  import tensorflowas tf
  mnist= input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)
  ```

- read_data_sets에 들어가 있는 디렉토리는 온라인에서 다운로드 받은 데이터를 임시로 저장해 놓을 위치

- ```python
  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  k = tf.matmul(x, W) + b
  y = tf.nn.softmax(k)
  ```

- x: training data를 저장하는 플레이스 홀더

- W: Weight

- b: bias 값

- model: y = tf.nn.softmax(tf.matmul(x, W) + b)

- tf.nn.softmax_cross_entropy_with_logits 함수는 softmax를 포함하고 있다. 그래서 softmax를 적용한 y를 넣으면 안 되고, softmax 적용 전인 k를 넣어야 한다.

- ```python
  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])
  learning_rate= 0.5
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = k))
  train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
  ```

- 모델을 정의했으면 학습을 위해서, 코스트 함수와 옵티마이저를 정의한다.

- ```python
  # 세션 초기화
  
  print ("Training")
  sess= tf.Session()
  init= tf.global_variables_initializer() #.run()
  sess.run(init)
  ```

- tf.Session() 를 이용해서 세션을 만들고, global_variable_initializer()를 이용해 변수들을 모두 초기화한다.

- 초기화 한 값은 sess.run에 넘겨서 세션을 초기화한다.

- ```python
  # Start training
  
  for _ in range(1000):
  # 1000번씩, 전체데이터에서 100개씩 뽑아서 트레이닝을 함.
  batch_xs, batch_ys= mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  ```

- 세션이 생성되었으면 이제 트레이닝을 시작한다.

- Batch training과 Stochastic training이 있다.

- Batch training: 학습을 할 때 전체 데이터를 가지고 한번에 학습을 하는 게 아니라 전체 데이터 셋을 몇 개로 쪼갠 후, 나눠서 트레이닝을 하는 방법

- 위 코드에 사용된 배치 트레이닝 방법은 "Stochastic training"

- 원칙대로라면 전체 55000개의 학습 데이터가 있기 때문에 배치 사이즈를100으로 했다면, 100개씩 550번 순차적으로 데이터를 읽어서 학습을 해야 한다.

- Stochastic training: 전체 데이터 중 일부를 샘플링해서 학습하는 방법!

  위 코드에서는 배치 한번에 100개씩의 데이터를 뽑아, 1000번 배치로 학습했다.

- 텐서 플로우 문서에 따르면, 전체 데이터를 순차적으로 학습 시키기에는 연산 비용이 비싸기 때문에, 샘플링을 해도 비슷한 정확도를 낼 수 있다는 근거로 예제 차원에서 간단하게 Stochastic training을 사용한 것으로 보인다.

  <br>

- ```python
  # 결과값 출력
  
  print('bis',sess.run(b))
  print('Wis',sess.run(W))
  ```

- ![결과값_1](https://user-images.githubusercontent.com/57430754/76479770-3bb37780-6450-11ea-83d9-8d2102201539.png)

- 먼저 앞에서 데이터를 로딩하도록 지정한 디렉토리에, 학습용 데이터를 다운 받아서 압축하는 것을 확인할 수 있다.

- ![결과값_2](https://user-images.githubusercontent.com/57430754/76479772-3ce4a480-6450-11ea-8ca0-c0748c974e8f.png)

- 그 다음 학습이 끝난 후, b, W 값이 출력되었다. W는 784 라인이기 때문에 중간을 생략하고 출력되었으나, 각 행을 모두 찍어보면 W 값이 들어가 있는 것을 볼 수 있다.

<br>

- ```python
  print ("Testing model")
  
  # Test trained model
  
  correct_prediction= tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print('accuracy ',sess.run(accuracy, feed_dict={x: mnist.test.images,
  y_: mnist.test.labels}))
  print ("done")
  ```

- 모델 검증

- mnist.test.image와 mnist.test.labels 데이터 셋을 이용하여 테스트를 진행

- 앞에 나온 모델에 mnist.test.image 데이터를 넣어서 예측을 한다.

- 그 결과를 mnist.test.labels(정답)과 비교해 정답률이 얼마나 되는지를 비교한다.

- correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))를 보면,

- tf.argmax 함수를 이해해야 한다. 

- argmax(y, 1)는 행렬 y에서 몇 번째에 가장 큰 값이 들어가 있는지를 리턴해주는 함수이다.

<br>

- ```python
  session = tf.InteractiveSession()
  
  data = tf.constant([9,2,11,4])
  idx= tf.argmax(data,0)
  
  print idx.eval()
  session.close()
  ```

- [9, 2, 11, 4]에서 최대수는 11이고, 이 위치는 2번째(0부터 시작)이기 때문에 0을 리턴한다.

- 두 번째 변수는 어느 축으로 카운트 할 것인지 선택한다. 1차원 배열의 경우에는 0을 사용한다.

- y: 2차원 행렬, 0이면 같은 "열"에서 최대값은 순서, 1이면 같은 "행"에서 최대값인 순서를 리턴한다.

- tf.argmax(y, 1)은 y의 각 행에서 가장 큰 값의 순서를 찾는다.

  y의 각 행을 0~9로 인식한 이미지의 확률을 가지고 있다.

- ![333](https://user-images.githubusercontent.com/57430754/76482108-951ea500-6456-11ea-98a6-c4bf88c15a7f.png)

- tf.argmax(y, 1)를 사용하면, 행별로 가장 큰 값을 리턴하기 때문에, 위 코드에서는 4가 리턴된다.

- 테스트용 데이터에서 원래 정답이 4로 되어 있다면, argmax(y_, 1)도 4를 리턴하기 때문에 tf.equal(tf.argmax(y,1),tf.argmax(y_,1))는 tf.equals(4,4)로 True를 리턴

- 모든 데이터 셋에서 검증을 하고 나서 그 결과에 True만 더해 전체 트레이닝 데이터 수로 나눠주면 결국 "정확도"가 나온다.

- tf.cast(boolean, tf.float32)를 하면 텐서플로우의 bool 값을 float32(실수)로 변환해준다.

- True는 1.0으로 False는 0.0으로 변환해준다.

- 이렇게 변환된 값들의 전체 평균을 구하면 되기 때문에, tf.reduce_mean을 사용한다.

<br>

- 여기서 tf.argmax(y, 1)을 사용하면, 행별로 가장 큰 값을 리턴하기 때문에 위의 값에서는 4가 리턴된다.
- 테스트용 데이터에서 원래 정답이 4로 되어 있다면 argmax(y_, 1)도 4를 리턴하기 때문에 tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))는 tf.equals(4, 4)로 True를 리턴하게 된다.
- 모든 테스트 셋에 대해서 검증을 하고 나서 그 결과에서 True만 더해서, 전체 트레이닝 데이터의 수로 나누어 주면 결국 정확도가 나온다.
- tf.cast(boolean, tf.float32)를 하면 텐서플로우의 bool 값을 float32(실수)로 변환해준다.
- True는 1.0으로 False는 0.0으로 변환해준다.
- 이렇게 변환된 값들의 전체 평균을 구하면 되기 때문에, tf.reduce_mean을 사용한다.

<br>

- 이렇게 정확도를 구하는 함수가 정의되었으면 이제 정확도를 구하기 위해 데이터를 넣는다.

- ```python
  # 정확도 구하기 위해 데이터 넣기
  
  sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
  ```

- x에 mnist.test.images 데이터 셋으로 이미지 데이터를 입력 받아서 y(예측 결과)를 계산

- y_에 minst.test.labels 정답을 입력 받아서, y와 y_로 정확도 accuracy를 구해서 출력

- 최종 출력된 accuracy 정확도는 0.9, 대략 90% 정도