# =============================================================================
#       Deep Learning models
# =============================================================================
        #this package is for some fatal errors when we use custom parameters in Random Search method
        from sklearn.externals.joblib import parallel_backend
        from keras import regularizers
        
        #DNN
        def DNN_create(layers_num,units,learning_rate,decay,dropout,l1,l2):
            model=Sequential()
            #bulid the layers
            for i in range(layers_num):
                if i==1:
                    model.add(layers.Dense(units,input_shape=(X_train.shape[1],),kernel_regularizer=regularizers.l1_l2(l1=l1,l2=l2)))
                else:
                    model.add(layers.Dense(units,kernel_regularizer=regularizers.l1_l2(l1=l1,l2=l2)))
                model.add(layers.Dropout(dropout))
                
            model.add(layers.Dense(1,activation='sigmoid'))
            
            optimizer=optimizers.RMSprop(lr=learning_rate,decay=decay)
            model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['acc'])

            return model
        
        # fix random seed for reproducibility
        np.random.seed(seed)
        
        model=KerasClassifier(build_fn=DNN_create,verbose=1)
        #Random Search for optimal hyper-paramters
        params_grid_DNN={'layers_num':[3,5,7],
                         'units':[16,32,64],
                         'learning_rate':[0.05,0.1,0.15],
                         'decay':[1e-6,1e-4,1e-2],
                         'dropout':[0.1,0.3,0.5],
                         'l1':[0.001,0.1,10],
                         'l2':[0.001,0.1,10],
                         'batch_size':[32,64,128],
                         'epochs':[20,40,80]}
        
        skf=StratifiedKFold(n_splits=cvfolds,shuffle=False,random_state=seed)
        cv_indices=skf.split(X_train,y_train)
        
        model=RandomizedSearchCV(model,params_grid_DNN,scoring='accuracy',n_jobs=corenum,cv=cv_indices,
                                 refit=True,n_iter=random_iter,random_state=seed)
        
        with parallel_backend('threading'):
            model.fit(X_train,y_train)
        
        #reshape the data for 3d input
        X_deep_matrix=np.append(X_train,X_test,axis=0)
        #transform the matrix data to tensor data
        X_deep_tensor=np.empty((X_deep_matrix.shape[0]-d+1,d,X_deep_matrix.shape[1]))
        
        for i in range(d-1,X_deep_matrix.shape[0]):
            X_deep_tensor[i-d+1]=X_deep_matrix[i-d+1:i+1,:]
        
        del X_deep_matrix
        #generate training and test sets for deep learning algorithm    
        X_train_deep=X_deep_tensor[:X_deep_tensor.shape[0]-test_set]
        X_test_deep=X_deep_tensor[X_deep_tensor.shape[0]-test_set:]
        
        del X_deep_tensor
        
        y_train_deep=y_train[d-1:len(y_train)]
        
        #RNN
        Model_Deep_RNN=Sequential()
        Model_Deep_RNN.add(layers.SimpleRNN(32,dropout=0.1,recurrent_dropout=0.1,return_sequences=True,
                                            input_shape=(X_train_deep.shape[1],X_train_deep.shape[2])))
        Model_Deep_RNN.add(layers.SimpleRNN(32,dropout=0.1,recurrent_dropout=0.1,return_sequences=True))
        Model_Deep_RNN.add(layers.SimpleRNN(32,dropout=0.1,recurrent_dropout=0.1,return_sequences=False))
        Model_Deep_RNN.add(layers.Dense(1,activation='sigmoid'))
        
        Model_Deep_RNN.compile(optimizer=RMSprop(),loss='binary_crossentropy',metrics=['accuracy'])
        Model_Deep_RNN.fit(X_train_deep,y_train_deep,steps_per_epoch=50,epochs=40)
        
        #bidirectional RNN
        Model_Deep_BRNN=Sequential()
        Model_Deep_BRNN.add(layers.Bidirectional(layers.SimpleRNN(32,dropout=0.1,recurrent_dropout=0.1,return_sequences=True,
                                             input_shape=(X_train_deep.shape[1],X_train_deep.shape[2]))))
        Model_Deep_BRNN.add(layers.Bidirectional(layers.SimpleRNN(32,dropout=0.1,recurrent_dropout=0.1,return_sequences=True)))
        Model_Deep_BRNN.add(layers.Bidirectional(layers.SimpleRNN(32,dropout=0.1,recurrent_dropout=0.1,return_sequences=False)))
        Model_Deep_BRNN.add(layers.Dense(1,activation='sigmoid'))
        
        Model_Deep_BRNN.compile(optimizer=RMSprop(),loss='binary_crossentropy',metrics=['accuracy'])
        Model_Deep_BRNN.fit(X_train_deep,y_train_deep,steps_per_epoch=50,epochs=40)
        
        #RNN-LSTM
        Model_Deep_RNNLSTM=Sequential()
        Model_Deep_RNNLSTM.add(layers.LSTM(32,dropout=0.1,recurrent_dropout=0.1,return_sequences=True,
                                            input_shape=(X_train_deep.shape[1],X_train_deep.shape[2])))
        Model_Deep_RNNLSTM.add(layers.LSTM(32,dropout=0.1,recurrent_dropout=0.1,return_sequences=True))
        Model_Deep_RNNLSTM.add(layers.LSTM(32,dropout=0.1,recurrent_dropout=0.1,return_sequences=False))
        Model_Deep_RNNLSTM.add(layers.Dense(1,activation='sigmoid'))
        
        Model_Deep_RNNLSTM.compile(optimizer=RMSprop(),loss='binary_crossentropy',metrics=['accuracy'])
        Model_Deep_RNNLSTM.fit(X_train_deep,y_train_deep,steps_per_epoch=50,epochs=40)
        
        #RNN-GRU
        Model_Deep_RNNGRU=Sequential()
        Model_Deep_RNNGRU.add(layers.GRU(32,dropout=0.1,recurrent_dropout=0.1,return_sequences=True,
                                            input_shape=(X_train_deep.shape[1],X_train_deep.shape[2])))
        Model_Deep_RNNGRU.add(layers.GRU(32,dropout=0.1,recurrent_dropout=0.1,return_sequences=True))
        Model_Deep_RNNGRU.add(layers.GRU(32,dropout=0.1,recurrent_dropout=0.1,return_sequences=False))
        Model_Deep_RNNGRU.add(layers.Dense(1,activation='sigmoid'))
        
        Model_Deep_RNNGRU.compile(optimizer=RMSprop(),loss='binary_crossentropy',metrics=['accuracy'])
        Model_Deep_RNNGRU.fit(X_train_deep,y_train_deep,steps_per_epoch=50,epochs=40)
