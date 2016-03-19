#ifndef QPHIX_COMM_H
#define QPHIX_COMM_H

#include <qmp.h>
#include <mpi.h>
#include <queue>

//DEBUG
#include <sstream>
//DEBUG


#warning "Using single sided communications"

namespace QPhiX
{

#define QPHIX_DSLASH_MPI_TAG (12)

	/*! Communicating version */
	template<typename T, int V, int S, const bool compressP>
	class Comms {
	public:
		Comms(Geometry<T,V,S,compressP>* geom)  
		{
			//get the underlying QMP communicator
			QMP_get_hidden_comm(QMP_comm_get_default(),reinterpret_cast<void**>(&mpi_comm_base));
#ifdef QPHIX_USE_SINGLE_SIDED_COMMS
			MPI_Info_create(&mpi_info);
#endif
			
			//topology information
			for(unsigned int i=0; i<4; i++){
				logical_dimensions[i]=QMP_get_logical_dimensions()[i];
				logical_coordinates[i]=QMP_get_logical_coordinates()[i];
			}
			
			// Deal with the faces
			NFaceDir[0] = (geom->Ny() * geom->Nz() * geom->Nt())/2;
			NFaceDir[1] = (geom->Nx() * geom->Nz() * geom->Nt())/2;
			NFaceDir[2] = (geom->Nx() * geom->Ny() * geom->Nt())/2;
			NFaceDir[3] = (geom->Nx() * geom->Ny() * geom->Nz())/2;
      

      
			// We have QMP
			// Decide which directions are local by appealing to 
			// QMP geometry
			if( QMP_get_logical_number_of_dimensions() != 4 ) {
				QMP_error("Number of QMP logical dimensions must be 4");
				QMP_abort(1);
			}


			for(int d = 0; d < 4; d++) {
				if( logical_dimensions[d] > 1 ){
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
      	  
			//init communicators, windows and buffers
			initBuffers();
      
			// Determine if I am minimum/maximum in the time direction in the processor grid:
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
      
		}

		~Comms(void)
		{
			for(int d = 0; d < 4; d++) {
				if(!localDir(d)) {
					
					//free buffers
					ALIGNED_FREE(sendToDir[2*d+0]);
					ALIGNED_FREE(sendToDir[2*d+1]);
#ifndef QPHIX_USE_SINGLE_SIDED_COMMS
					ALIGNED_FREE(recvFromDir[2*d+0]);
					ALIGNED_FREE(recvFromDir[2*d+1]);
#else
					//unlock windows
					//unlockDir(2*d+0);
					//unlockDir(2*d+1);
					
					//free windows
					MPI_Win_free(&winDir[2*d+0]);
					MPI_Win_free(&winDir[2*d+1]);
					
					//Free info param
					MPI_Info_free(&mpi_info);
#endif
				}
			}
		}

		inline void startSendDir(int d) {
			/* **** MPI HERE ******* */
			if (  MPI_Isend( (void *)sendToDir[d], faceInBytes[d/2], MPI_BYTE, myNeighboursInDir[d],  QPHIX_DSLASH_MPI_TAG, *mpi_comm_base, &reqSendToDir[d] ) != MPI_SUCCESS ) { 
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
			if ( MPI_Irecv((void *)recvFromDir[d], faceInBytes[d/2], MPI_BYTE, myNeighboursInDir[d], QPHIX_DSLASH_MPI_TAG, *mpi_comm_base, &reqRecvFromDir[d]) != MPI_SUCCESS ) { 
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
		
		//test if sent/received is completed
		inline bool testSendToDir(int d){
			int iflag;
			if( MPI_Test(reqSendToDir[d], &iflag, NULL) != MPI_SUCCESS){
				QMP_error("Wait on recv from dir failed\n");
				QMP_abort(1);
			}
			return static_cast<bool>(iflag);
		}
		
		inline bool testRecvFromDir(int d){
			int iflag;
			if( MPI_Test(reqRecvFromDir[d], &iflag, NULL) != MPI_SUCCESS){
				QMP_error("Wait on recv from dir failed\n");
				QMP_abort(1);
			}
			return static_cast<bool>(iflag);
		}
		
		
		
		//single sided only routines
#ifdef QPHIX_USE_SINGLE_SIDED_COMMS
		inline int oppDir(int d){
			return d+1-2*(d%2);
		}
		
		inline void lockDir(int d){
			if (MPI_Win_lock_all(0, winDir[d]) != MPI_SUCCESS){
				QMP_error("Win-lock failed!\n");
				QMP_abort(1);
			}
		}
		
		inline void unlockDir(int d){
			if (MPI_Win_unlock_all(winDir[d]) != MPI_SUCCESS){
				QMP_error("Win-unlock failed!\n");
				QMP_abort(1);
			}
		}
		
		inline void initPutDir(int d){
			if (MPI_Win_fence(MPI_NOPRECEDE, winDir[d]) != MPI_SUCCESS){
				QMP_error("Init Put failed!\n");
				QMP_abort(1);
			}
		}
		
		inline void finishPutDir(int d){
			if( MPI_Win_fence(MPI_NOSUCCEED, winDir[d]) != MPI_SUCCESS){
				QMP_error("Finish Put failed!\n");
				QMP_abort(1);
			}
		}
		
		inline void executePutDir(int d){
			if (MPI_Put(reinterpret_cast<void*>(sendToDir[d]),faceInBytes[d/2],MPI_BYTE,myNeighboursInDir[d],0,faceInBytes[d/2],MPI_BYTE,winDir[d]) != MPI_SUCCESS){
				QMP_error("Put failed!\n");
				QMP_abort(1);
			}
		}
#endif


		inline void progressComms() {
			int flag = 0;
			MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, *mpi_base_comm, &flag, MPI_STATUS_IGNORE);
		}

		//debug
		inline   int  getMyRank() const { return myRank; }
		inline   int  getFaceSize(int d) const { return faceInBytes[d]; }
		//debug
		
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
		queue<int> queue; //communication queue

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
		int logical_dimensions[4];
		int logical_coordinates[4];
		
		//communicators
		MPI_Comm* mpi_comm_base;
#ifdef QPHIX_USE_SINGLE_SIDED_COMMS
		MPI_Win winDir[8];
		MPI_Info mpi_info;
#endif
		
		
		void initBuffers(){
			
			for(int d = 0; d < 4; d++) {
				if(!localDir(d)) {
					
					//the sendTo buffers can simply be allocated, as they do not need to be in some kind of window:
					sendToDir[2*d + 0]   = (T*)ALIGNED_MALLOC(faceInBytes[d], 4096);
					sendToDir[2*d + 1]   = (T*)ALIGNED_MALLOC(faceInBytes[d], 4096);
					//recvFromDir[2*d + 0]   = (T*)ALIGNED_MALLOC(faceInBytes[d], 4096);
					//recvFromDir[2*d + 1]   = (T*)ALIGNED_MALLOC(faceInBytes[d], 4096);
										
					//do forward and backward windows: the buffers needs to be swapped relative to the windows, as passing data to 2*d+1
					//will make the data in 2*d+0 available.
					MPI_Win_allocate(faceInBytes[d], 1, mpi_info, *mpi_comm_base, reinterpret_cast<void**>(&recvFromDir[2*d + 1]), &winDir[2*d + 0]);
					MPI_Win_allocate(faceInBytes[d], 1, mpi_info, *mpi_comm_base, reinterpret_cast<void**>(&recvFromDir[2*d + 0]), &winDir[2*d + 1]);
					
					//lock dirs:
					//lockDir(2*d+0);
					//lockDir(2*d+1);
				}
				else {
					//set dummy buffers
					sendToDir[2*d+0]   = NULL;
					sendToDir[2*d+1]   = NULL;
					recvFromDir[2*d+0] = NULL;
					recvFromDir[2*d+1] = NULL;
					
					//set dummy windows
					winDir[2*d+0]=MPI_WIN_NULL;
					winDir[2*d+1]=MPI_WIN_NULL;
				}// End if local dir
			} // End loop over dir	
		}
	};

	namespace CommsUtils {
		void sumDouble(double* d) { QMP_sum_double(d); };
		void sumDoubleArray(double *d, int n) { QMP_sum_double_array(d,n); }
		int numNodes() { return QMP_get_number_of_nodes(); }
	};

}; // Namespace 


#endif
