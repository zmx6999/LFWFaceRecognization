from sklearn import datasets,metrics,model_selection,decomposition,svm,pipeline
import matplotlib.pyplot as plt
lfw_people = datasets.fetch_lfw_people(min_faces_per_person=70,resize=.4)
n_samples,h,w = lfw_people.images.shape
print(n_samples)
x = lfw_people.images.reshape((n_samples,-1))
n_features = x.shape[1]
print(n_features)
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = len(target_names)
print(n_classes)
x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,test_size=.25,random_state=42)

pca = decomposition.PCA(n_components=150,whiten=True,svd_solver='randomized')
svc = svm.SVC(kernel='rbf',class_weight='balanced')
estimator = pipeline.Pipeline(steps=[('pca',pca),('svc',svc)])
grid_search = model_selection.GridSearchCV(estimator,{'svc__C':[1e3,5e3,1e4,5e4,1e5],'svc__gamma':[0.0001,0.0005,0.001,0.005,0.01,0.1]})
grid_search.fit(x_train,y_train)
print(grid_search.score(x_test,y_test))
y_pred = grid_search.predict(x_test)
eigenfaces = grid_search.best_estimator_.named_steps['pca'].components_
print(metrics.classification_report(y_test,y_pred,target_names=target_names))
print(metrics.confusion_matrix(y_test,y_pred,labels=range(n_classes)))

def title(y_test,y_pred,target_names,i):
    test_name = target_names[y_test[i]].rsplit(' ',1)[-1]
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    return 'true:%s\npred:%s'%(test_name,pred_name)

def gallery(images,titles,h,w,n_cols=4,n_rows=3):
    plt.figure(figsize=(1.8*n_cols,2.4*n_rows))
    plt.subplots_adjust(top=.9,left=.01,right=.99,bottom=0,hspace=.24)
    for i in range(n_cols*n_rows):
        plt.subplot(n_rows,n_cols,i+1)
        plt.imshow(images[i].reshape((h,w)),cmap=plt.cm.gray)
        plt.title(titles[i])
        plt.xticks(())
        plt.yticks(())
    plt.show()

titles = [title(y_test,y_pred,target_names,i) for i in range(len(x_test))]
gallery(x_test,titles,h,w)
eigenface_titles = ['eigenface%s'%i for i in range(len(eigenfaces))]
gallery(eigenfaces,eigenface_titles,h,w)