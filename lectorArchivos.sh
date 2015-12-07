#! /bin/bash

SRC=$1
DEST=$2
if [ ! -d "$1" ] && [ ! -d "$2" ]; then
	echo "Necesitan ser directorios"
else
	#Creacion de archivos
	ls -R "${SRC}" | awk '
	/:$/&&f{s=$0;f=0}
	/:$/&&!f{sub(/:$/,"");s=$0;f=1;next}
	NF&&f{ print s"/"$0 }' | grep -iE 'jpeg|jpg|png|bmp|tiff|tif' > src.txt

	ls -R "${DEST}" | awk '
	/:$/&&f{s=$0;f=0}
	/:$/&&!f{sub(/:$/,"");s=$0;f=1;next}
	NF&&f{ print s"/"$0 }' | grep -iE 'jpeg|jpg|png|bmp|tiff|tif' > dest.txt

	#Lectura de archivos
	  while read -r linea
	    do	     
	    #sobre el segundo directorio
	    imagen1=${linea##*/}
	    ext1=${imagen1##*.}
	    nomRep=false
	    copiar=false
	        while read -r linea2
	        do
	        	imagen2=${linea2##*/}
	        	ext2=${imagen2##*.}
	        	#Manda a llamar a CUDA
	        	#Si nom iguales & pixels iguales -> No se copia
	        	#Si nom iguales & pixels dif -> Se renombra origen
	        	#Si nom dif & pixels iguales -> No se copia
	        	#Si nom dif & pixels dif -> Se copia

	        	#Comparamos pixeles
	        	#echo "Comp: "$linea" con "$linea2
	        	./proyecto $linea $linea2
	        	resultado=$?
	        	if [ "$resultado" = 1 ]; then
	        		copiar=true
	        		##Son diferentes a nivel pixel
	        		#Nombres iguales?
		        	if [ ${imagen1} = ${imagen2} ]; then       		
		        		nomRep=true
		        	fi
		        else
		        	#Son iguales a nivel pixel
		        	copiar=false
		        	nomRep=false
		        	break
	        	fi
	        	
	        done < "dest.txt"
	        
	        if $copiar ; then      	
	        	if $nomRep ; then		        
			        #Cambiar el nombre
			        dateC=$(date +%d%m%Y)
					timeC=$(date +%H%M)
					destino=$DEST"/"$dateC$timeC"."$ext1		
					cp ${linea} ${destino}
					
					echo $linea" copiada a "$DEST"/"$dateC$timeC"."$ext1 >> log.txt
				else
					cp ${linea} ${DEST}
					echo $linea" copiada a"$DEST >> log.txt
		        fi		        
	        fi
	    done < "src.txt"
fi