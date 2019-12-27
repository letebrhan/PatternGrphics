import pandas as pd
from scipy import stats
import numpy as np
import statistics as s
from statsmodels import robust
#Session1 1 data analysis
Session3=pd.read_csv('Session3/Session3.csv')
## after deleting all the cross and Explanation
#Session3_part2=pd.read_csv('Session3/Session3_part2.csv')

#Concatnate the two separated file of Experiment 1
Ses_conca=[Session3]
Ses_All_3=pd.concat(Ses_conca,axis=0)
#delete user 16
Ses_All_File_except16=Ses_All_3.Users!='16_User'
Ses_All_File_3=Ses_All_3[Ses_All_File_except16]
Ses_All_File_3.to_csv('Exported_Dataset_Ses_3/Ses_All_File_3.csv',index=False)

Ses_All_File_3.to_csv('Exported_Dataset_Ses_3/Ses_All_File_3.csv',index=False)
Media_1=Ses_All_File_3.MEDIA_NAME=='Media1'
Slide_1=Ses_All_File_3[Media_1]
Media_2=Ses_All_File_3.MEDIA_NAME=='Media2'
Slide_2=Ses_All_File_3[Media_2]
Media_3=Ses_All_File_3.MEDIA_NAME=='Media3'
Slide_3=Ses_All_File_3[Media_3]
Media_4=Ses_All_File_3.MEDIA_NAME=='Media4'
Slide_4=Ses_All_File_3[Media_4]
Media_5=Ses_All_File_3.MEDIA_NAME=='Media5'
Slide_5=Ses_All_File_3[Media_5]
Media_6=Ses_All_File_3.MEDIA_NAME=='Media6'
Slide_6=Ses_All_File_3[Media_6]
Media_7=Ses_All_File_3.MEDIA_NAME=='Media7'
Slide_7=Ses_All_File_3[Media_7]
Media_8=Ses_All_File_3.MEDIA_NAME=='Media8'
Slide_8=Ses_All_File_3[Media_8]
Media_9=Ses_All_File_3.MEDIA_NAME=='Media9'
Slide_9=Ses_All_File_3[Media_9]
Media_10=Ses_All_File_3.MEDIA_NAME=='Media10'
Slide_10=Ses_All_File_3[Media_10]

Slides_mod3=[Slide_1,Slide_2,Slide_3,Slide_4,Slide_5,Slide_6,
          Slide_7,Slide_8,Slide_9,Slide_10]
Slides1_Ses_3=pd.concat(Slides_mod3,axis=0)
Slides1_Ses_3.to_csv('Exported_Dataset_Ses_3/Slides1_Ses_3.csv',index=False)

#-------Locate only fixation data-------------
Fix_file_Ses=Slides1_Ses_3.USER=='U_fix'
Fix_Ses1=Slides1_Ses_3[Fix_file_Ses]
Fix_file_final_Ses3=Fix_Ses1.loc[:,['Users','MEDIA_NAME','FPOGX','FPOGY']]
Fix_file_final_Ses3.to_csv('Exported_Dataset_Ses_3/Fix_file_final_Ses3.csv',index=False)

b1=1/3
b2=2/3
#-----list of tester names in array
list_of_users=['1_User','2_User','3_User','4_User','5_User','6_User','7_User','8_User','9_User','10_User','11_User','12_User','13_User',
               '14_User','15_User','17_User','18_User','19_User','20_User','21_User','22_User','23_User','24_User','25_User','26_User'
               ,'27_User','28_User','29_User','30_User','31_User','32_User']
#------Creating column names to the  new features obtained from the area of the media or image---------------
new_col=['Testers','Media','A1_fix_No','A2_fix_No','A3_fix_No','A4_fix_No','A5_fix_No','A6_fix_No','total_noFixation','Scan_path']
temp_df1=pd.DataFrame(columns=new_col)
temp_df2=pd.DataFrame(columns=new_col)
temp_df3=pd.DataFrame(columns=new_col)
temp_df4=pd.DataFrame(columns=new_col)
temp_df5=pd.DataFrame(columns=new_col)
temp_df6=pd.DataFrame(columns=new_col)
temp_df7=pd.DataFrame(columns=new_col)
temp_df8=pd.DataFrame(columns=new_col)
temp_df9=pd.DataFrame(columns=new_col)
temp_df10=pd.DataFrame(columns=new_col)
#---- iterating the number of testers times for each media--------
for i in range(0,31):  
    #iterate for each use i in the fix file
    user_data=Fix_file_final_Ses3.Users==list_of_users[i]
    users=Fix_file_final_Ses3[user_data]
    #-----division of image1 into 6 area and count the number of fixation in each area-----
    c1=[0,0,0,0,0,0]
    media1_data=users.MEDIA_NAME=='Media1'
    Data1=users[media1_data]
    #---- removing the first line which contains the first fixation
    data1_remov_fix1=Data1.iloc[1:,]
    Data_X1=data1_remov_fix1.loc[:,'FPOGX']  
    Data_Y1=data1_remov_fix1.loc[:,'FPOGY']
    #print(data1)
    D1=0    #......the average distance among all consecutive fixation points
    nextPoint1=0   # the next fixation point for the given media
    for point in range(len(Data_X1)):
        if((66/1920)<Data_X1.values[point]) and (Data_X1.values[point]<=(662/1920)) and ((0)<Data_Y1.values[point]) and (Data_Y1.values[point]<=(0.5)):
            c1[0]+=1
        elif((662/1920)<Data_X1.values[point]) and (Data_X1.values[point]<=(1258/1920)) and((0)<Data_Y1.values[point]) and (Data_Y1.values[point]<=(0.5)):
            a=[Data_X1.values[point],Data_Y1.values[point]]
            c1[1]+=1
        elif((1258/1920)<Data_X1.values[point]) and (Data_X1.values[point]<=(1854/1920))and ((0)<Data_Y1.values[point]) and (Data_Y1.values[point]<=(0.5)):
            c1[2]+=1
        elif((66/1920)<Data_X1.values[point]) and (Data_X1.values[point]<=(662/1920))and (0.5<Data_Y1.values[point]) and (Data_Y1.values[point]<=(1076/1080)):
            c1[3]+=1
        elif((662/1920)<Data_X1.values[point]) and (Data_X1.values[point]<=(1258/1920))and (0.5<Data_Y1.values[point]) and (Data_Y1.values[point]<=(1076/1080)):
            c1[4]+=1
        elif((1258/1920)<Data_X1.values[point]) and (Data_X1.values[point]<=(1858/1920)) and (0.5<Data_Y1.values[point]) and (Data_Y1.values[point]<=(1076/1080)):
            c1[5]+=1
        if (point<len(Data_X1)-1):
            nextPoint1=point+1
            D_increment_X1=Data_X1.values[nextPoint1]
            D_increment_Y1=Data_Y1.values[nextPoint1]
            D1=D1+np.sqrt((np.square((Data_X1.values[point])-D_increment_X1))+(np.square((Data_Y1.values[point])-D_increment_Y1)))
    #---- appending rows fixations values for each area in media1 to the current dataframe----
    df1={'Testers':list_of_users[i],
         'Media':'Media1','A1_fix_No':c1[0],'A2_fix_No':c1[1],'A3_fix_No':c1[2],'A4_fix_No':c1[3],'A5_fix_No':c1[4],
                          'A6_fix_No':c1[5],'total_noFixation':len(Data_X1),'Scan_path':D1,'Saccade_scanPath':D1/(len(Data_X1)-1)}
    temp_df1=temp_df1.append(df1,ignore_index=True, verify_integrity=False)       
    #---------- division of image2 into 6 area and count the number of fixation in each area-----
    c2=[0,0,0,0,0,0]      
    media2_data=users.MEDIA_NAME=='Media2'
    Data2=users[media2_data]
    #---- removing the first line which contains the first fixation
    data2_remov_fix1=Data2.iloc[1:,]
    Data_X2=data2_remov_fix1.loc[:,'FPOGX']      
    Data_Y2=data2_remov_fix1.loc[:,'FPOGY']
    D2=0
    nextPoint2=0

    for point2 in range(len(Data_X2)):
        if (((210/1920)<Data_X2.values[point2]) and (Data_X2.values[point2]<=(710/1920))) and ((0<Data_Y2.values[point2]) and (Data_Y2.values[point2]<=(0.5))):#A1
            c2[0]+=1
        elif((710/1920)<Data_X2.values[point2]) and (Data_X2.values[point2]<=(1210/1920)) and(0<Data_Y2.values[point2]) and (Data_Y2.values[point2]<=(0.5)):#A2
            c2[1]+=1
        elif((1210/1920)<Data_X2.values[point2]) and (Data_X2.values[point2]<=(1710/1920))and (0<Data_Y2.values[point2]) and (Data_Y2.values[point2]<=(0.5)):#A3
            c2[2]+=1
        elif((210/1920)<Data_X2.values[point2]) and (Data_X2.values[point2]<=(710/1920))and (0.5<Data_Y2.values[point2]) and (Data_Y2.values[point2]<=(1)):#A4
            c2[3]+=1
        elif((710/1920)<Data_X2.values[point2]) and (Data_X2.values[point2]<=(1210/1920))and (0.5<Data_Y2.values[point2]) and (Data_Y2.values[point2]<=(1)):#A5
            c2[4]+=1
        elif((1210/1920)<Data_X2.values[point2]) and (Data_X2.values[point2]<=(1710/1920)) and (0.5<Data_Y2.values[point2]) and (Data_Y2.values[point2]<=(1)):#A6
            c2[5]+=1
        if (point2<len(Data_X2)-1):
            nextPoint2=point2+1
            D_increment_X2=Data_X2.values[nextPoint2]
            D_increment_Y2=Data_Y2.values[nextPoint2]
            D2=D2+np.sqrt((np.square((Data_X2.values[point2])-D_increment_X2))+(np.square((Data_Y2.values[point2])-D_increment_Y2)))

    #---- appending rows fixations values for each area in media2 to the current dataframe----
    df2={'Testers':list_of_users[i],'Media':'Media2','A1_fix_No':c2[0],'A2_fix_No':c2[1],'A3_fix_No':c2[2],'A4_fix_No':c2[2],'A5_fix_No':c2[4],
                          'A6_fix_No':c2[5],'total_noFixation':len(Data_X2),'Scan_path':D2,'Saccade_scanPath':D2/(len(Data_X2)-1)}
    temp_df2=temp_df2.append(df2,ignore_index=True, verify_integrity=False)
    #---------- division of image3 into 6 area and count the number of fixation in each area-----
    c3=[0,0,0,0,0,0]
    media3_data=users.MEDIA_NAME=='Media3'
    Data3=users[media3_data]
    #---- removing the first line which contains the first fixation
    data3_remov_fix1=Data3.iloc[1:,]
    Data_X3=data3_remov_fix1.loc[:,'FPOGX']  
    Data_Y3=data3_remov_fix1.loc[:,'FPOGY']
    D3=0
    nextPoint3=0
    for point3 in range(len(Data_X3)):
        if (((0)<Data_X3.values[point3]) and (Data_X3.values[point3]<=(b1))) and (((4/1080)<Data_Y3.values[point3]) and (Data_Y3.values[point3]<=(0.5))):#A1
            c3[0]+=1
        elif((b1)<Data_X3.values[point3]) and (Data_X3.values[point3]<=(b2)) and((4/1080)<Data_Y3.values[point3]) and (Data_Y3.values[point3]<=(0.5)):#A2
            c3[1]+=1
        elif((b2)<Data_X3.values[point3]) and (Data_X3.values[point3]<=(1))and ((4/1080)<Data_Y3.values[point3]) and (Data_Y3.values[point3]<=(0.5)):#A3
            c3[2]+=1
        elif((0)<Data_X3.values[point3]) and (Data_X3.values[point3]<=(b1))and (0.5<Data_Y3.values[point3]) and (Data_Y3.values[point3]<=(1076/1080)):#A4
            c3[3]+=1
        elif((b1)<Data_X3.values[point3]) and (Data_X3.values[point3]<=(b2))and (0.5<Data_Y3.values[point3]) and (Data_Y3.values[point3]<=(1076/1080)):#A5
            c3[4]+=1
        elif((b2)<Data_X3.values[point3]) and (Data_X3.values[point3]<=(1)) and (0.5<Data_Y3.values[point3]) and (Data_Y3.values[point3]<=(1076/1080)):#A6
            c3[5]+=1
        if (point3<len(Data_X3)-1):
            nextPoint3=point3+1
            D_increment_X3=Data_X3.values[nextPoint3]
            D_increment_Y3=Data_Y3.values[nextPoint3]
            D3=D3+np.sqrt((np.square((Data_X3.values[point3])-D_increment_X3))+(np.square((Data_Y3.values[point3])-D_increment_Y3)))

    #---- appending rows fixations values for each area in media3 to the current dataframe----
    df3={'Testers':list_of_users[i],'Media':'Media3','A1_fix_No':c3[0],'A2_fix_No':c3[1],'A3_fix_No':c3[2],'A4_fix_No':c3[3],'A5_fix_No':c3[4],
                          'A6_fix_No':c3[5],'total_noFixation':len(Data_X3),'Scan_path':D3,'Saccade_scanPath':D3/(len(Data_X3)-1)}
    temp_df3=temp_df3.append(df3,ignore_index=True, verify_integrity=False)
    #---------- division of image4 into 6 area and count the number of fixation in each area-----
    c4=[0,0,0,0,0,0]
    media4_data=users.MEDIA_NAME=='Media4'
    Data4=users[media4_data]
    #---- removing the first line which contains the first fixation
    data4_remov_fix1=Data4.iloc[1:,]
    Data_X4=data4_remov_fix1.loc[:,'FPOGX']  
    Data_Y4=data4_remov_fix1.loc[:,'FPOGY']   
    D4=0
    nextPoint4=0
    for point4 in range(len(Data_X4)):
        if (((275/1920)<Data_X4.values[point4]) and (Data_X4.values[point4]<=(732/1920))) and ((0<Data_Y4.values[point4]) and (Data_Y4.values[point4]<=(0.5))):#A1
            c4[0]+=1
        elif((732/1920)<Data_X4.values[point4]) and (Data_X4.values[point4]<=(1188/1920)) and(0<Data_Y4.values[point4]) and (Data_Y4.values[point4]<=(0.5)):#A2
            a=[Data_X4.values[point4],Data_Y4.values[point4]]
            c4[1]+=1
        elif((1188/1920)<Data_X4.values[point4]) and (Data_X4.values[point4]<=(1645/1920))and ((0)<Data_Y4.values[point4]) and (Data_Y4.values[point4]<=(0.5)):#A3
            c4[2]+=1
        elif((275/1920)<Data_X4.values[point4]) and (Data_X4.values[point4]<=(732/1920))and (0.5<Data_Y4.values[point4]) and (Data_Y4.values[point4]<=(1)):#A4
            c4[3]+=1
        elif((732/1920)<Data_X4.values[point4]) and (Data_X4.values[point4]<=(1188/1920))and (0.5<Data_Y4.values[point4]) and (Data_Y4.values[point4]<=(1)):#A5
            c4[4]+=1
        elif((1188/1920)<Data_X4.values[point4]) and (Data_X4.values[point4]<=(1645/1920)) and (0.5<Data_Y4.values[point4]) and (Data_Y4.values[point4]<=(1)):#A6
            c4[5]+=1
        if (point4<len(Data_X4)-1):
            nextPoint4=point4+1
            D_increment_X4=Data_X4.values[nextPoint4]
            D_increment_Y4=Data_Y4.values[nextPoint4]
            D4=D4+np.sqrt((np.square((Data_X4.values[point4])-D_increment_X4))+(np.square((Data_Y4.values[point4])-D_increment_Y4)))

    #---- appending rows fixations values for each area in media4 to the current dataframe----
    df4={'Testers':list_of_users[i],'Media':'Media4','A1_fix_No':c4[0],'A2_fix_No':c4[1],'A3_fix_No':c4[2],'A4_fix_No':c4[3],'A5_fix_No':c4[4],
                          'A6_fix_No':c4[5],'total_noFixation':len(Data_X4),'Scan_path':D4,'Saccade_scanPath':D4/(len(Data_X4)-1)}
    temp_df4=temp_df4.append(df4,ignore_index=True, verify_integrity=False)
    #-- division of image5 into 6 area and count the number of fixation in each area-----
    c5=[0,0,0,0,0,0]
    media5_data=users.MEDIA_NAME=='Media5'
    Data5=users[media5_data]
    #---- removing the first line which contains the first fixation
    data5_remov_fix1=Data5.iloc[1:,]
    Data_X5=data5_remov_fix1.loc[:,'FPOGX']  
    Data_Y5=data5_remov_fix1.loc[:,'FPOGY'] 
    D5=0
    nextPoint5=0
    for point5 in range(len(Data_X5)):
        if (((90/1920)<Data_X5.values[point5]) and (Data_X5.values[point5]<=(670/1920))) and ((0<Data_Y5.values[point5]) and (Data_Y5.values[point5]<=(0.5))):#A1
            c5[0]+=1
        elif((670/1920)<Data_X5.values[point5]) and (Data_X5.values[point5]<=(1250/1920)) and(0<Data_Y5.values[point5]) and (Data_Y5.values[point5]<=(0.5)):#A2
            a=[Data_X5.values[point5],Data_Y5.values[point5]]
            c5[1]+=1
        elif((1250/1920)<Data_X5.values[point5]) and (Data_X5.values[point5]<=(1830/1920))and ((0)<Data_Y5.values[point5]) and (Data_Y5.values[point5]<=(0.5)):#A3
            c5[2]+=1
        elif((90/1920)<Data_X5.values[point5]) and (Data_X5.values[point5]<=(670/1920))and (0.5<Data_Y5.values[point5]) and (Data_Y5.values[point5]<=(1)):#A4
            c5[3]+=1
        elif((670/1920)<Data_X5.values[point5]) and (Data_X5.values[point5]<=(1250/1920))and (0.5<Data_Y5.values[point5]) and (Data_Y5.values[point5]<=(1)):#A5
            c5[4]+=1
        elif((1250/1920)<Data_X5.values[point5]) and (Data_X5.values[point5]<=(1830/1920)) and (0.5<Data_Y5.values[point5]) and (Data_Y5.values[point5]<=(1)):#A6
            c5[5]+=1
        if (point5<len(Data_X5)-1):
            nextPoint5=point5+1
            D_increment_X5=Data_X5.values[nextPoint5]
            D_increment_Y5=Data_Y5.values[nextPoint5]
            D5=D5+np.sqrt((np.square((Data_X5.values[point5])-D_increment_X5))+(np.square((Data_Y5.values[point5])-D_increment_Y5)))

    #---- appending rows fixations values for each area in media5 to the current dataframe----
    df5={'Testers':list_of_users[i],'Media':'Media5','A1_fix_No':c5[0],'A2_fix_No':c5[1],'A3_fix_No':c5[2],'A4_fix_No':c5[3],'A5_fix_No':c5[4],
                          'A6_fix_No':c5[5],'total_noFixation':len(Data_X5),'Scan_path':D5,'Saccade_scanPath':D5/(len(Data_X5)-1)}
    temp_df5=temp_df5.append(df5,ignore_index=True, verify_integrity=False)
    #---------- division of image6 into 6 area and count the number of fixation in each area-----
    c6=[0,0,0,0,0,0]
    media6_data=users.MEDIA_NAME=='Media6'
    Data6=users[media6_data]
    #---- removing the first line which contains the first fixation
    data6_remov_fix1=Data6.iloc[1:,]
    Data_X6=data6_remov_fix1.loc[:,'FPOGX']  
    Data_Y6=data6_remov_fix1.loc[:,'FPOGY'] 
    D6=0
    nextPoint6=0
    for point6 in range(len(Data_X6)):
        if (((14/1920)<Data_X6.values[point6]) and (Data_X6.values[point6]<=(645/1920))) and ((0<Data_Y6.values[point6]) and (Data_Y6.values[point6]<=(0.5))):#A1
            c6[0]+=1
        elif((645/1920)<Data_X6.values[point6]) and (Data_X6.values[point6]<=(1275/1920)) and(0<Data_Y6.values[point6]) and (Data_Y6.values[point6]<=(0.5)):#A2
            a=[Data_X6.values[point6],Data_Y6.values[point6]]
            c6[1]+=1
        elif((1275/1920)<Data_X6.values[point6]) and (Data_X6.values[point6]<=(1906/1920))and (0<Data_Y6.values[point6]) and (Data_Y6.values[point6]<=(0.5)):#A3
            c6[2]+=1
        elif((275/1920)<Data_X6.values[point6]) and (Data_X6.values[point6]<=(645/1920))and (0.5<Data_Y6.values[point6]) and (Data_Y6.values[point6]<=(1)):#A4
            c6[3]+=1
        elif((645/1920)<Data_X6.values[point6]) and (Data_X6.values[point6]<=(1275/1920))and (0.5<Data_Y6.values[point6]) and (Data_Y6.values[point6]<=(1)):#A5
            c6[4]+=1
        elif((1275/1920)<Data_X6.values[point6]) and (Data_X6.values[point6]<=(1906/1920)) and (0.5<Data_Y6.values[point6]) and (Data_Y6.values[point6]<=(1)):#A6
            c6[5]+=1
        if (point6<len(Data_X6)-1):
            nextPoint6=point6+1
            D_increment_X6=Data_X6.values[nextPoint6]
            D_increment_Y6=Data_Y6.values[nextPoint6]
            D6=D6+np.sqrt((np.square((Data_X6.values[point6])-D_increment_X6))+(np.square((Data_Y6.values[point6])-D_increment_Y6)))
    #---- appending rows fixations values for each area in media6 to the current dataframe----
    df6={'Testers':list_of_users[i],'Media':'Media6','A1_fix_No':c6[0],'A2_fix_No':c6[1],'A3_fix_No':c6[2],'A4_fix_No':c6[3],'A5_fix_No':c6[4],
                          'A6_fix_No':c6[5],'total_noFixation':len(Data_X6),'Scan_path':D6,'Saccade_scanPath':D6/(len(Data_X6)-1)}
    temp_df6=temp_df6.append(df6,ignore_index=True, verify_integrity=False)
    #---------- division of image7 into 6 area and count the number of fixation in each area-----
    c7=[0,0,0,0,0,0]
    media7_data=users.MEDIA_NAME=='Media7'
    Data7=users[media7_data]
    #---- removing the first line which contains the first fixation
    data7_remov_fix1=Data7.iloc[1:,]
    Data_X7=data7_remov_fix1.loc[:,'FPOGX']  
    Data_Y7=data7_remov_fix1.loc[:,'FPOGY'] 
    D7=0
    nextPoint7=0
    for point7 in range(len(Data_X7)):
        if(0<Data_X7.values[point7]) and (Data_X7.values[point7]<=b1) and ((26/1080)<Data_Y7.values[point7]) and (Data_Y7.values[point7]<=(0.5)):
            c7[0]+=1
        elif(b1<Data_X7.values[point7]) and (Data_X7.values[point7]<=b2) and((26/1080)<Data_Y7.values[point7]) and (Data_Y7.values[point7]<=(0.5)):
            c7[1]+=1
        elif(b2<Data_X7.values[point7]) and (Data_X7.values[point7]<=1)and ((26/1080)<Data_Y7.values[point7]) and (Data_Y7.values[point7]<=(0.5)):
            c7[2]+=1
        elif(0<Data_X7.values[point7]) and (Data_X7.values[point7]<=b1)and (0.5<Data_Y7.values[point7]) and (Data_Y7.values[point7]<=(1054/1080)):
            c7[3]+=1
        elif(b1<Data_X7.values[point7]) and (Data_X7.values[point7]<=b2)and (0.5<Data_Y7.values[point7]) and (Data_Y7.values[point7]<=(1054/1080)):
            c7[4]+=1
        elif(b2<Data_X7.values[point7]) and (Data_X7.values[point7]<=1) and (0.5<Data_Y7.values[point7]) and (Data_Y7.values[point7]<=(1054/1080)):
            c7[5]+=1

        if (point7<len(Data_X7)-1):
            nextPoint7=point7+1
            D_increment_X7=Data_X7.values[nextPoint7]
            D_increment_Y7=Data_Y7.values[nextPoint7]
            D7=D7+np.sqrt((np.square((Data_X7.values[point7])-D_increment_X7))+(np.square((Data_Y7.values[point7])-D_increment_Y7)))

    #---- appending rows fixations values for each area in media7 to the current dataframe----
    df7={'Testers':list_of_users[i],'Media':'Media7','A1_fix_No':c7[0],'A2_fix_No':c7[1],'A3_fix_No':c7[2],'A4_fix_No':c7[3],'A5_fix_No':c7[4],
                          'A6_fix_No':c7[5],'total_noFixation':len(Data_X7),'Scan_path':D7,'Saccade_scanPath':D7/(len(Data_X7)-1)}
    temp_df7=temp_df7.append(df7,ignore_index=True, verify_integrity=False) 
    #---------- division of image8 into 6 area and count the number of fixation in each area-----
    c8=[0,0,0,0,0,0]
    media8_data=users.MEDIA_NAME=='Media8'
    Data8=users[media8_data]
    #---- removing the first line which contains the first fixation
    data8_remov_fix1=Data8.iloc[1:,]
    Data_X8=data8_remov_fix1.loc[:,'FPOGX']  
    Data_Y8=data8_remov_fix1.loc[:,'FPOGY'] 
    D8=0
    nextPoint8=0
    for point8 in range(len(Data_X8)):
        if (((24/1920)<Data_X8.values[point8]) and (Data_X8.values[point8]<=(648/1920))) and ((0<Data_Y8.values[point8]) and (Data_Y8.values[point8]<=(0.5))):#A1
            c8[0]+=1
        elif((648/1920)<Data_X8.values[point8]) and (Data_X8.values[point8]<=(1272/1920)) and(0<Data_Y8.values[point8]) and (Data_Y8.values[point8]<=(0.5)):#A2
            a=[Data_X8.values[point8],Data_Y8.values[point8]]
            c8[1]+=1
        elif((1272/1920)<Data_X8.values[point8]) and (Data_X8.values[point8]<=(1896/1920))and (0<Data_Y8.values[point8]) and (Data_Y8.values[point8]<=(0.5)):#A3
            c8[2]+=1
        elif((24/1920)<Data_X8.values[point8]) and (Data_X8.values[point8]<=(648/1920))and (0.5<Data_Y8.values[point8]) and (Data_Y8.values[point8]<=(1)):#A4
            c8[3]+=1
        elif((648/1920)<Data_X8.values[point8]) and (Data_X8.values[point8]<=(1272/1920))and (0.5<Data_Y8.values[point8]) and (Data_Y8.values[point8]<=(1)):#A5
            c8[4]+=1
        elif((1272/1920)<Data_X8.values[point8]) and (Data_X8.values[point8]<=(1896/1920)) and (0.5<Data_Y8.values[point8]) and (Data_Y8.values[point8]<=(1)):#A6
            c8[5]+=1
        if (point8<len(Data_X8)-1):
            nextPoint8=point8+1
            D_increment_X8=Data_X8.values[nextPoint8]
            D_increment_Y8=Data_Y8.values[nextPoint8]
            D8=D8+np.sqrt((np.square((Data_X8.values[point8])-D_increment_X8))+(np.square((Data_Y8.values[point8])-D_increment_Y8)))

    #---- appending rows fixations values for each area in media8 to the current dataframe----
    df8={'Testers':list_of_users[i],'Media':'Media8','A1_fix_No':c8[0],'A2_fix_No':c8[1],'A3_fix_No':c8[2],'A4_fix_No':c8[3],'A5_fix_No':c8[4],
                          'A6_fix_No':c8[5],'total_noFixation':len(Data_X8),'Scan_path':D8,'Saccade_scanPath':D8/(len(Data_X8)-1)}
    temp_df8=temp_df8.append(df8,ignore_index=True, verify_integrity=False) 
    #---------- division of image9 into 6 area and count the number of fixation in each area-----
    c9=[0,0,0,0,0,0]
    media9_data=users.MEDIA_NAME=='Media9'
    Data9=users[media9_data]
    #---- removing the first line which contains the first fixation
    data9_remov_fix1=Data9.iloc[1:,]
    Data_X9=data9_remov_fix1.loc[:,'FPOGX']  
    Data_Y9=data9_remov_fix1.loc[:,'FPOGY'] 
    D9=0
    nextPoint9=0
    for point9 in range(len(Data_X9)):
        if (((72/1920)<Data_X9.values[point9]) and (Data_X9.values[point9]<=(664/1920))) and ((0<Data_Y9.values[point9]) and (Data_Y9.values[point9]<=(0.5))):#A1
            c9[0]+=1
        elif((664/1920)<Data_X9.values[point9]) and (Data_X9.values[point9]<=(1256/1920)) and(0<Data_Y9.values[point9]) and (Data_Y9.values[point9]<=(0.5)):#A2
            c9[1]+=1
        elif((1256/1920)<Data_X9.values[point9]) and (Data_X9.values[point9]<=(1848/1920))and (0<Data_Y9.values[point9]) and (Data_Y9.values[point9]<=(0.5)):#A3
            c9[2]+=1
        elif((72/1920)<Data_X9.values[point9]) and (Data_X9.values[point9]<=(664/1920))and (0.5<Data_Y9.values[point9]) and (Data_Y9.values[point9]<=(1)):#A4
            c9[3]+=1
        elif((664/1920)<Data_X9.values[point9]) and (Data_X9.values[point9]<=(1256/1920))and (0.5<Data_Y9.values[point9]) and (Data_Y9.values[point9]<=(1)):#A5
            c9[4]+=1
        elif((1256/1920)<Data_X9.values[point9]) and (Data_X9.values[point9]<=(1848/1920)) and (0.5<Data_Y9.values[point9]) and (Data_Y9.values[point9]<=(1)):#A6
            c9[5]+=1
        if (point9<len(Data_X9)-1):
            nextPoint9=point9+1
            D_increment_X9=Data_X9.values[nextPoint9]
            D_increment_Y9=Data_Y9.values[nextPoint9]
            D9=D9+np.sqrt((np.square((Data_X9.values[point9])-D_increment_X9))+(np.square((Data_Y9.values[point9])-D_increment_Y9)))

    #---- appending rows fixations values for each area in media9 to the current dataframe----
    df9={'Testers':list_of_users[i],'Media':'Media9','A1_fix_No':c9[0],'A2_fix_No':c9[1],'A3_fix_No':c9[2],'A4_fix_No':c9[3],'A5_fix_No':c9[4],
                          'A6_fix_No':c9[5],'total_noFixation':len(Data_X9),'Scan_path':D9,'Saccade_scanPath':D9/(len(Data_X9)-1)}
    temp_df9=temp_df9.append(df9,ignore_index=True, verify_integrity=False)
    #---------- division of image10 into 6 area and count the number of fixation in each area-----
    c10=[0,0,0,0,0,0]
    media10_data=users.MEDIA_NAME=='Media10'
    Data10=users[media10_data]
    #---- removing the first line which contains the first fixation
    data10_remov_fix1=Data10.iloc[1:,]
    Data_X10=data10_remov_fix1.loc[:,'FPOGX']  
    Data_Y10=data10_remov_fix1.loc[:,'FPOGY'] 
    D10=0
    nextPoint10=0

    for point10 in range(len(Data_X10)):
        if (((70/1920)<Data_X10.values[point10]) and (Data_X10.values[point10]<=(663/1920))) and ((0<Data_Y10.values[point10]) and (Data_Y10.values[point10]<=(0.5))):#A1
            c10[0]+=1
        elif((663/1920)<Data_X10.values[point10]) and (Data_X10.values[point10]<=(1257/1920)) and(0<Data_Y10.values[point10]) and (Data_Y10.values[point10]<=(0.5)):#A2
            a=[Data_X10.values[point10],Data_Y10.values[point10]]
            c10[1]+=1
        elif((1257/1920)<Data_X10.values[point10]) and (Data_X10.values[point10]<=(1850/1920))and (0<Data_Y10.values[point10]) and (Data_Y10.values[point10]<=(0.5)):#A3
            c10[2]+=1
        elif((70/1920)<Data_X10.values[point10]) and (Data_X10.values[point10]<=(663/1920))and (0.5<Data_Y10.values[point10]) and (Data_Y10.values[point10]<=(1)):#A4
            c10[3]+=1
        elif((663/1920)<Data_X10.values[point10]) and (Data_X10.values[point10]<=(1257/1920))and (0.5<Data_Y10.values[point10]) and (Data_Y10.values[point10]<=(1)):#A5
            c10[4]+=1
        elif((1257/1920)<Data_X10.values[point10]) and (Data_X10.values[point10]<=(1850/1920)) and (0.5<Data_Y10.values[point10]) and (Data_Y10.values[point10]<=(1)):#A6
            c10[5]+=1
        if (point10<len(Data_X10)-1):
            nextPoint10=point10+1
            D_increment_X10=Data_X10.values[nextPoint10]
            D_increment_Y10=Data_Y10.values[nextPoint10]
            D10=D10+np.sqrt((np.square((Data_X10.values[point10])-D_increment_X10))+(np.square((Data_Y10.values[point10])-D_increment_Y10)))

    #---- appending rows fixations values for each area in media10 to the current dataframe----
    df10={'Testers':list_of_users[i],'Media':'Media10','A1_fix_No':c10[0],'A2_fix_No':c10[1],'A3_fix_No':c10[2],'A4_fix_No':c10[3],'A5_fix_No':c10[4],
                          'A6_fix_No':c10[5],'total_noFixation':len(Data_X10),'Scan_path':D10,'Saccade_scanPath':D10/(len(Data_X10)-1)}
    temp_df10=temp_df10.append(df10,ignore_index=True, verify_integrity=False)                        
df=[temp_df1,temp_df2,temp_df3,temp_df4,temp_df5,temp_df6,temp_df7,temp_df8,temp_df9,temp_df10]
df_fixation_Area=pd.concat(df,axis=0)

#locating and Selecting the 2 LPD and RPD features  
Ses1_LRPD=Slides1_Ses_3.loc[:,['Users','USER','MEDIA_NAME','LPD','RPD']]
#-----fileter the gaze data only
LRPD=Ses1_LRPD.USER=='U_gaze'
LRPD_gaze=Ses1_LRPD[LRPD]
LRPD_gaze.to_csv('Exported_Dataset_Ses_3/LRPD_gaze.csv',index=False)

#locating and Selecting the 2 LPD and RPD features  
Ses1_LRFS=Slides1_Ses_3.loc[:,['Users','USER','MEDIA_NAME','FPOGD','LPS','RPS']]
#-----fileter the fixation data only
LRFS=Ses1_LRFS.USER=='U_fix'
LRFS_fix=Ses1_LRFS[LRFS]
LRFS_fix.to_csv('Exported_Dataset_Ses_3/LRFS_fix.csv',index=False)

# ... lists of media used......
list_of_media=['Media1','Media2','Media3','Media4','Media5','Media6','Media7','Media8','Media9','Media10']
#.... colomn names on the whole for the features of FPOG(Fixation Point of Gaze), LPS (Left Pupil Scale factor) and RPS (Right Pupil Scale factor)

flr_col=['Testers_flr','Medias_flr',
                   
                   'fpogDuration_sum','fpogd_min','fpogd_max',
                   'fpogd_mean','fpogd_mad','fpogd_median','fpogd_skew','fpogd_var','fpogd_std',
                   'fpogd_hmean','fpogd_gmean','fpogd_range','fpogd_iqr','fpogd_first_quar','fpogd_thrid_quar','fpogd_kurtosis',
                   
                   'lps_sum','lps_min','lps_max',
                   'lps_mean','lps_mad','lps_median','lps_skew','lps_var','lps_std',
                   'lps_hmean','lps_gmean','lps_range','lps_iqr','lps_first_quar','lps_thrid_quar', 'lps_kurtosis',
                   
                   'rps_sum','rps_min','rps_max','rps_mean','rps_mad','rps_median','rps_skew','rps_var','rps_std',
                   'rps_hmean','rps_gmean','rps_range','rps_iqr','rps_first_quar','rps_thrid_quar','rps_kurtosis']
#.... colomn names on the whole for the features of LPD(Left Pupil Diameter) and RPD (Right Pupil Diameter)

lrpd_col=['Testers_lrpd','Medias_lrpd',
                   'lpd_sum','lpd_min','lpd_max','lpd_mean','lpd_mad',
                   'lpd_median','lpd_skew','lpd_var','lpd_std','lpd_hmean','lpd_gmean','lpd_range','lpd_iqr',
                   'lpd_first_quar','lpd_thrid_quar', 'lpd_kurtosis',
                   'rpd_sum','rpd_min','rpd_max','rpd_mean','rpd_mad','rpd_median','rpd_skew','rpd_var','rpd_std',
                   'rpd_hmean','rpd_gmean','rpd_range','rpd_iqr','rpd_first_quar','rpd_thrid_quar', 'rpd_kurtosis']
# declaration of the flr and lpd_rpd variables and intilizing them to an empty array which latter will store the descriptive statistics values of
# flr stands for FPOGD, LPS and RPS and lpd_rpd stands for the the RPD and LPD ... then applying to the the descriptive statistics
flr=[]
lpd_rpd=[]
temp_lrpd=pd.DataFrame(columns=lrpd_col)
temp_flr=pd.DataFrame(columns=flr_col)

for k in range(0,10):
    # for each k image do the following
    for j in range(0,31):        
        #----for each j user for the Gaze files of LPD and RPD pupil diameter
        lrpd_data=LRPD_gaze.Users==list_of_users[j]
        lrpd_users=LRPD_gaze[lrpd_data]
        #--- for each k media for the Gaze files of LPD and RPD pupil diameter---
        lrpd_media=lrpd_users.MEDIA_NAME==list_of_media[k]
        lrpd=lrpd_users[lrpd_media]
        #---- removing the first line which contains the first fixation
        lrpd_fix=lrpd.iloc[1:,]
        #----locating LPD and RPD----
        lpd_data=lrpd_fix.loc[:,'LPD']
        rpd_data=lrpd_fix.loc[:,'RPD']
           #print(robust.mad(lpd_data))
        lpd_rpd.append({'Testers_lrpd':list_of_users[j],'Medias_lrpd':list_of_media[k],
                   'lpd_sum':sum(lpd_data),'lpd_min':min(lpd_data),'lpd_max':max(lpd_data),'lpd_mean':np.mean(lpd_data),'lpd_mad':robust.mad(lpd_data),
                   'lpd_median':np.median(lpd_data),'lpd_skew':stats.skew(lpd_data),'lpd_var':np.var(lpd_data),'lpd_std':np.std(lpd_data),
                   'lpd_hmean':s.harmonic_mean(lpd_data),'lpd_gmean':stats.gmean(lpd_data),'lpd_range':np.ptp(lpd_data),'lpd_iqr':stats.iqr(lpd_data),
                   'lpd_first_quar':np.percentile(lpd_data,25),'lpd_thrid_quar':np.percentile(lpd_data,75), 'lpd_kurtosis':stats.kurtosis(lpd_data),
                   
                   'rpd_sum':sum(rpd_data),'rpd_min':min(rpd_data),'rpd_max':max(rpd_data),'rpd_mean':np.mean(rpd_data),'rpd_mad':robust.mad(rpd_data),
                   'rpd_median':np.median(rpd_data),'rpd_skew':stats.skew(rpd_data),'rpd_var':np.var(rpd_data),'rpd_std':np.std(rpd_data),
                   'rpd_hmean':s.harmonic_mean(rpd_data),'rpd_gmean':stats.gmean(rpd_data),'rpd_range':np.ptp(rpd_data),'rpd_iqr':stats.iqr(rpd_data),
                   'rpd_first_quar':np.percentile(rpd_data,25),'rpd_thrid_quar':np.percentile(rpd_data,75), 'rpd_kurtosis':stats.kurtosis(rpd_data)
                   })
        #----for each j user for the fixation files of FPOGD, LPS and RPS
        flr_data=LRFS_fix.Users==list_of_users[j]
        flr_users=LRFS_fix[flr_data]
        #--- for each k media for the fixation files of FPOGD, LPS and RPS---
        flr_media=flr_users.MEDIA_NAME==list_of_media[k]
        flr_for_itr=flr_users[flr_media]
        #---- removing the first line which contains the first fixation
        flr_fix=flr_for_itr.iloc[1:,]
        #----locating the scan path-----
        #scanPath=flr_fix.loc[:,'Scan_path']
        
        #----locating LPD and RPD----
        fpogd_data=flr_fix.loc[:,'FPOGD']      
        lps_data=flr_fix.loc[:,'LPS']
        rps_data=flr_fix.loc[:,'RPS']
        
        flr.append({'Testers_flr':list_of_users[j],'Medias_flr':list_of_media[k],
                   
                   'fpogDuration_sum':sum(fpogd_data),'fpogd_min':min(fpogd_data),'fpogd_max':max(fpogd_data),
                   'fpogd_mean':np.mean(fpogd_data),'fpogd_mad':robust.mad(fpogd_data),
                   'fpogd_median':np.median(fpogd_data),'fpogd_skew':stats.skew(fpogd_data),'fpogd_var':np.var(fpogd_data),'fpogd_std':np.std(fpogd_data),
                   'fpogd_hmean':s.harmonic_mean(fpogd_data),'fpogd_gmean':stats.gmean(fpogd_data),'fpogd_range':np.ptp(fpogd_data),'fpogd_iqr':stats.iqr(fpogd_data),
                   'fpogd_first_quar':np.percentile(fpogd_data,25),'fpogd_thrid_quar':np.percentile(fpogd_data,75), 'fpogd_kurtosis':stats.kurtosis(fpogd_data),
                   
                   'lps_sum':sum(lps_data),'lps_min':min(lps_data),'lps_max':max(lps_data),'lps_mean':np.mean(lps_data),'lps_mad':robust.mad(lps_data),
                   'lps_median':np.median(lps_data),'lps_skew':stats.skew(lps_data),'lps_var':np.var(lps_data),'lps_std':np.std(lps_data),
                   'lps_hmean':s.harmonic_mean(lps_data),'lps_gmean':stats.gmean(lps_data),'lps_range':np.ptp(lps_data),'lps_iqr':stats.iqr(lps_data),
                   'lps_first_quar':np.percentile(lps_data,25),'lps_thrid_quar':np.percentile(lps_data,75), 'lps_kurtosis':stats.kurtosis(lps_data),
                   
                   'rps_sum':sum(rps_data),'rps_min':min(rps_data),'rps_max':max(rps_data),'rps_mean':np.mean(rps_data),'rps_mad':robust.mad(rps_data),
                   'rps_median':np.median(rps_data),'rps_skew':stats.skew(rps_data),'rps_var':np.var(rps_data),'rps_std':np.std(rps_data),
                   'rps_hmean':s.harmonic_mean(rps_data),'rps_gmean':stats.gmean(rps_data),'rps_range':np.ptp(rps_data),'rps_iqr':stats.iqr(rps_data),
                   'rps_first_quar':np.percentile(rps_data,25),'rps_thrid_quar':np.percentile(rps_data,75), 'rps_kurtosis':stats.kurtosis(rps_data)
                   })
# the values in array stored into the dataframe as follow(temp_lrpd and temp_flr are decleared above as empty dataframe)        
stats_lrpd_data=temp_lrpd.append(lpd_rpd,ignore_index=True, verify_integrity=False)  
stats_flr_data=temp_flr.append(flr,ignore_index=True, verify_integrity=False)   
  
##      writing the result of the fixation in each area fix, LRPD gaze and FLRS fix data

df_fixation_Area.to_csv('Exported_Dataset_Ses_3/df_fixation_Area.csv',index=False)
stats_lrpd_data.to_csv('Exported_Dataset_Ses_3/stats_lrpd_data.csv',index=False)
stats_flr_data.to_csv('Exported_Dataset_Ses_3/stats_flr_data.csv',index=False)

fixation_Area=pd.read_csv('Exported_Dataset_Ses_3/df_fixation_Area.csv')
lrpd_data=pd.read_csv('Exported_Dataset_Ses_3/stats_lrpd_data.csv')
flr_data=pd.read_csv('Exported_Dataset_Ses_3/stats_flr_data.csv')

Data1=[fixation_Area,lrpd_data,flr_data]
Ses_Data_1=pd.concat(Data1, axis=1)
Ses_Data_1.to_csv('Exported_Dataset_Ses_3/Ses_Data_3.csv',index=False)
All_DS_Ses_3=pd.read_csv('Exported_Dataset_Ses_3/Ses_Data_3.csv')
All_DS_Ses_3.drop(['Testers_lrpd','Medias_lrpd','Testers_flr','Medias_flr'], axis=1, inplace=True)
#All_dataStat_Ses_1.drop(['DataType','MEDIA_NAME_FLR','UsersName_FLR','DataType_FLR'], axis=1, inplace=True)
All_DS_Ses_3.to_csv('Exported_Dataset_Ses_3/All_DS_Ses_3.csv',index=False)