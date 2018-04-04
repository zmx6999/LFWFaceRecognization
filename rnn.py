from sklearn import datasets,model_selection
from keras import utils,models,layers,optimizers
lfw_people = datasets.fetch_lfw_people(min_faces_per_person=70,resize=.4)
n_samples,h,w = lfw_people.images.shape
x = lfw_people.images.reshape(n_samples,h,w)/255.0
target_names = lfw_people.target_names
n_class = len(target_names)
y = utils.to_categorical(lfw_people.target,n_class)
x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,test_size=.1,random_state=42)

sequential = models.Sequential()
sequential.add(layers.GRU(64,input_shape=(h,w),dropout=.25))
#sequential.add(layers.LSTM(64,input_shape=(h,w),dropout=.25))
#sequential.add(layers.LSTM(128,input_shape=(h,w),dropout=.25))
sequential.add(layers.Dense(n_class,activation='softmax'))

sequential.compile(optimizer=optimizers.adam(),loss='categorical_crossentropy',metrics=['accuracy'])
sequential.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=120)