#include <stdio.h>      /* printf, fgets */
#include <stdlib.h>     /* atoi */

void setup() {
  Serial.begin(115200);
}

char buffer[3];
String Data="";
void loop() {
   if (Serial.available() >= 3) {
    for(int i=0; i<3;i++)
    {
     buffer[i]=Serial.read(); 
     Data+=buffer[i];
    }
    int height = Data.toInt()-100;
    Serial.println(height);
    Data="";
  }
}
 
