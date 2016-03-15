#ifndef QPHIX_COMM_H
#define QPHIX_COMM_H

#include <qmp.h>
#include <mpi.h>


namespace QPhiX
{



#define QPHIX_DSLASH_MPI_TAG (12)

	/*! Communicating version */
	template<typename T, int V, int S, const bool compressP>
	class Comms {
	public:
		Comms(Geometry<T,V,S,compressP>* geom)  
		{
			// Deal with the faces
			NFaceDir[0] = (geom->Ny() * geom->Nz() * geom->Nt())/2;
			NFaceDir[1] = (geom->Nx() * geom->Nz() * geom->Nt())/2;
			NFaceDir[2] = (geom->Nx() * geom->Ny() * geom->Nt())/2;
			NFaceDir[3] = (geom->Nx() * geom->Ny() * geom->Nz())/2;
      

      
			// We have QMP
			// Decide which directions are local by appealing to 
			// QMP geometry
			const int* machine_size = QMP_get_logical_dimensions();
			if( QMP_get_logical_number_of_dimensions() != 4 ) {
				QMP_error("Number of QMP logical dimensions must be 4");
				QMP_abort(1);
			}


      
			for(int d = 0; d < 4; d++) {
				if( machine_size[d] > 1 ){
					localDir_[d] = false;
				}
				else {
					localDir_[d] = true;
				}
			}
			myRank = QMP_get_node_number();
			const int *qmp_dims = QMP_get_logical_dimensions();
			const int  *qmp_coords = QMP_get_logical_coordinates();
			int fw_neigh_coords[4];
			int bw_neigh_coords[4];
      
			totalBufSize = 0;
			for(int d = 0; d < 4; d++) {
				if ( !localDir(d) ) {
					for(int dim=0; dim < 4; dim++) { 
						fw_neigh_coords[dim]=bw_neigh_coords[dim]=qmp_coords[dim];
					}
					bw_neigh_coords[d]--; if (bw_neigh_coords[d] < 0  ) bw_neigh_coords[d] = qmp_dims[d]-1;
					fw_neigh_coords[d]++; if (fw_neigh_coords[d] == qmp_dims[d] ) fw_neigh_coords[d] = 0;
					myNeighboursInDir[2*d+0] = QMP_get_node_number_from( bw_neigh_coords);
					myNeighboursInDir[2*d+1] = QMP_get_node_number_from( fw_neigh_coords);
					faceInBytes[d] = NFaceDir[d]*12*sizeof(T); //12 T elements of the half spinor
					totalBufSize += faceInBytes[d];
				}
				else {
					myNeighboursInDir[2*d+0]= myRank;
					myNeighboursInDir[2*d+1]= myRank;
					faceInBytes[d] = 0;
				}
			}
			totalBufSize *= 4; // 2 bufs for sends & 2 for recvs
      
			for(int d = 0; d < 4; d++) {
				if(!localDir(d)) {
					sendToDir[2*d+0]   = (T*)ALIGNED_MALLOC(faceInBytes[d], 4096);
					sendToDir[2*d+1]   = (T*)ALIGNED_MALLOC(faceInBytes[d], 4096);
					recvFromDir[2*d+0] = (T*)ALIGNED_MALLOC(faceInBytes[d], 4096);
					recvFromDir[2*d+1] = (T*)ALIGNED_MALLOC(faceInBytes[d], 4096);
	  
					msgmem_sendToDir[2*d+0] = QMP_declare_msgmem(sendToDir[2*d+0], faceInBytes[d]);
					msgmem_sendToDir[2*d+1] = QMP_declare_msgmem(sendToDir[2*d+1], faceInBytes[d]);
					msgmem_recvFromDir[2*d+0] = QMP_declare_msgmem(recvFromDir[2*d+0], faceInBytes[d]);
					msgmem_recvFromDir[2*d+1] = QMP_declare_msgmem(recvFromDir[2*d+1], faceInBytes[d]);
	  
					mh_sendToDir[2*d+1] = QMP_declare_send_to(msgmem_sendToDir[2*d+1], myNeighboursInDir[2*d+1], 0);
					mh_recvFromDir[2*d+0] = QMP_declare_receive_from(msgmem_recvFromDir[2*d+0], myNeighboursInDir[2*d+0], 0);		
					mh_sendToDir[2*d+0] = QMP_declare_send_to(msgmem_sendToDir[2*d+0], myNeighboursInDir[2*d+0], 0);
					mh_recvFromDir[2*d+1] = QMP_declare_receive_from(msgmem_recvFromDir[2*d+1], myNeighboursInDir[2*d+1], 0);
				}
				else {
					sendToDir[2*d+0]   = NULL;
					sendToDir[2*d+1]   = NULL;
					recvFromDir[2*d+0] = NULL;
					recvFromDir[2*d+1] = NULL;

					msgmem_sendToDir[2*d+0] = NULL;
					msgmem_sendToDir[2*d+1] = NULL;
					msgmem_recvFromDir[2*d+0] = NULL;
					msgmem_recvFromDir[2*d+1] = NULL;
	  
					mh_sendToDir[2*d+1] = NULL;
					mh_recvFromDir[2*d+0] = NULL;
					mh_sendToDir[2*d+0] = NULL;
					mh_recvFromDir[2*d+1] = NULL;
				}
			} // End loop over dir
      
			// Determine if I am minimum/maximum in the time direction in the processor grid:
			const int* logical_dimensions = QMP_get_logical_dimensions();
			const int* logical_coordinates = QMP_get_logical_coordinates();
      
			amIPtMin_ = (logical_coordinates[3] == 0);
			amIPtMax_ = (logical_coordinates[3] == (logical_dimensions[3]-1) );

			numNonLocalDir_ = 0;
			// Count the number of non local dirs
			// and keep a compact map
			for(int d=0; d < 4; d++) { 
				if ( ! localDir(d) ) { 
					nonLocalDir_[ numNonLocalDir_ ] = d;
					numNonLocalDir_++;
				}
			}
    		initComms();
      
		}

		~Comms(void)
		{
			for(int d = 0; d < 4; d++) {
				if(!localDir(d)) {
					ALIGNED_FREE(sendToDir[2*d+0]);
					ALIGNED_FREE(sendToDir[2*d+1]);
					ALIGNED_FREE(recvFromDir[2*d+0]);
					ALIGNED_FREE(recvFromDir[2*d+1]);
					MPI_Comm_free(commDir[2*d+0]);
					MPI_Comm_free(commDir[2*d+1]);
				}
			}
		}


		inline void startSendDir(int d) {
			/* **** MPI HERE ******* */
			if (  MPI_Isend( (void *)sendToDir[d], faceInBytes[d/2], MPI_BYTE, myNeighboursInDir[d],  QPHIX_DSLASH_MPI_TAG, MPI_COMM_WORLD, &reqSendToDir[d] ) != MPI_SUCCESS ) { 
				QMP_error("Failed to start send in forward T direction\n");
				QMP_abort(1);
			}

#ifdef QMP_DIAGNOSTICS
			printf("My Rank: %d, start send dir %d,  My Records: srce=%d, dest=%d len=%d\n", myRank, d, myRank, myNeighboursInDir[d], faceInBytes[d/2]);
#endif
		}
    
		inline void finishSendDir(int d) {
#ifdef QMP_DIAGNOSTICS
			printf("My Rank: %d, finish send dir %d,  My Records: srce=%d, dest=%d len=%d\n", myRank, d, myRank, myNeighboursInDir[d], faceInBytes[d/2]);
#endif

			/* **** MPI HERE ******* */
			if (  MPI_Wait(&reqSendToDir[d], MPI_STATUS_IGNORE) != MPI_SUCCESS ) { 
				QMP_error("Wait on send failed \n");
				QMP_abort(1);
			}
		}

		inline void startRecvFromDir(int d) { 
			/* **** MPI HERE ******* */
			if ( MPI_Irecv((void *)recvFromDir[d], faceInBytes[d/2], MPI_BYTE, myNeighboursInDir[d], QPHIX_DSLASH_MPI_TAG, MPI_COMM_WORLD, &reqRecvFromDir[d]) != MPI_SUCCESS ) { 
				QMP_error("Recv from dir failed\n");
				QMP_abort(1);
			}

#ifdef QMP_DIAGNOSTICS
			printf("My Rank: %d, start recv from dir %d,  My Records: srce=%d, dest=%d len=%d\n", myRank, d, myNeighboursInDir[d], myRank,  faceInBytes[d/2]);
#endif
		}

		inline void finishRecvFromDir(int d) {
#ifdef QMP_DIAGNOSTICS
			printf("My Rank: %d, finish recv from dir %d,  My Records: srce=%d, dest=%d len=%d\n", myRank, d, myNeighboursInDir[d], myRank,  faceInBytes[d/2]);
#endif
			/* **** MPI HERE ******* */
			if ( MPI_Wait(&reqRecvFromDir[d], MPI_STATUS_IGNORE) != QMP_SUCCESS ) { 
				QMP_error("Wait on recv from dir failed\n");
				QMP_abort(1);
			}
		}

		inline void progressComms() {
			int flag = 0;
			MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
		}

		inline   bool localX() const { return localDir_[0]; }
		inline   bool localY() const { return localDir_[1]; }

		inline   bool localZ() const { return localDir_[2]; }
		inline   bool localT() const { return localDir_[3]; }
		inline   bool localDir(int d) const { return localDir_[d]; }


		inline T* getSendToDirBuf(int dir) { return sendToDir[dir]; }
		inline T* getRecvFromDirBuf(int dir) { return recvFromDir[dir]; }

		/* Am I the processor with smallest (t=0) in time */
		inline bool amIPtMin() const 
		{
			return amIPtMin_;
		}

		/* Am I the process with the largest (t=P_t - 1)  coordinate in time */ 
		inline bool amIPtMax() const 
		{
			return amIPtMax_;
		}

    
		T* sendToDir[8]; // Send Buffers
		T* recvFromDir[8]; // Recv Buffers

	private:
    
		// Ranks of the neighbours in the Y, Z and T directions
		int myRank;
		int myNeighboursInDir[8];
    
		unsigned int faceInBytes[4];
		size_t totalBufSize;
		//  Hack for Karthik here: (Handles for the requests) 
		MPI_Request reqSendToDir[8];
		MPI_Request reqRecvFromDir[8];

		int numNonLocalDir() { return numNonLocalDir_; }
		int nonLocalDir(int d)  { return nonLocalDir_[d]; }

		int NFaceDir[4];
		bool localDir_[4];
		bool amIPtMin_;
		bool amIPtMax_;
		int numNonLocalDir_;
		int nonLocalDir_[4];
		
		//communicators
		MPI_Comm commDir[8];

		void initComms(){
			
			//some global variables
			MPI_Comm mpi_comm_tmp;
			
			//get logical coordinates
			const int* logical_dimensions = QMP_get_logical_dimensions();
			const int* logical_coordinates = QMP_get_logical_coordinates();
			
			for(int d = 0; d < 4; d++) {
				if(!localDir(d)) {
					//get some params:
					int ldim=logical_dimensions[d];
					int lcoord=logical_coordinates[d];
					
					//key is simply my rank
					int key=myRank;
					
					
					//even first
					int color=(lcoord%2==0 ? lcoord/2 : ((lcoord+ldim-1)%ldim)/2);

					MPI_Comm_split(*mpi_base_comm, color, key, &mpi_comm_tmp);
						
					//decide whether I created up-comm or down-comm
					if(lcoord%2==0){
						//this is the comm in plus direction
						commDir[2*d + 1]=mpi_comm_tmp;
					}
					else{
						//this is the comm in minus direction
						commDir[2*d + 0]=mpi_comm_tmp;
					}
					
					
					//odd comes next
					int color=(lcoord%2==1 ? ((lcoord+1)%ldim)/2 : lcoord/2);
					
					MPI_Comm_split(*mpi_base_comm, color, key, &mpi_comm_tmp);
					
					//decide whether I created up-comm or down-comm
					if(lcoord%2==1){
						//this is the comm in plus direction
						commDir[2*d + 1]=mpi_comm_tmp;
					}
					else{
						//this is the comm in minus direction
						commDir[2*d + 0]=mpi_comm_tmp;
					}
				}
				else{
					commDir[2*d + 0]=MPI_COMM_NULL;
					commDir[2*d + 1]=MPI_COMM_NULL;
				}
			}
		}
		
	};

	namespace CommsUtils {
		void sumDouble(double* d) { QMP_sum_double(d); };
		void sumDoubleArray(double *d, int n) { QMP_sum_double_array(d,n); }
		int numNodes() { return QMP_get_number_of_nodes(); }
	};

}; // Namespace 


#endif
