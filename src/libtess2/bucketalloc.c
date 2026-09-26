/*
** SGI FREE SOFTWARE LICENSE B (Version 2.0, Sept. 18, 2008) 
** Copyright (C) [dates of first publication] Silicon Graphics, Inc.
** All Rights Reserved.
**
** Permission is hereby granted, free of charge, to any person obtaining a copy
** of this software and associated documentation files (the "Software"), to deal
** in the Software without restriction, including without limitation the rights
** to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
** of the Software, and to permit persons to whom the Software is furnished to do so,
** subject to the following conditions:
** 
** The above copyright notice including the dates of first publication and either this
** permission notice or a reference to http://oss.sgi.com/projects/FreeB/ shall be
** included in all copies or substantial portions of the Software. 
**
** THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
** INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
** PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL SILICON GRAPHICS, INC.
** BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
** TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
** OR OTHER DEALINGS IN THE SOFTWARE.
** 
** Except as contained in this notice, the name of Silicon Graphics, Inc. shall not
** be used in advertising or otherwise to promote the sale, use or other dealings in
** this Software without prior written authorization from Silicon Graphics, Inc.
*/
/*
** Author: Mikko Mononen, July 2009.
*/

#include <stdio.h>
#include <stdlib.h>
#include "tesselator.h"

//#define CHECK_BOUNDS

typedef struct BucketAlloc BucketAlloc;
typedef struct Bucket Bucket;

struct Bucket
{
	Bucket *next;
};

struct BucketAlloc
{
	void *freelist;		// Items returned with bucketFree() (reused first, LIFO).
	Bucket *buckets;
	unsigned char *cur;	// Next never-used item in the newest bucket.
	unsigned char *end;	// End of the newest bucket.
	unsigned int itemSize;
	unsigned int bucketSize;
	const char *name;
	TESSalloc* alloc;
};

static int CreateBucket( struct BucketAlloc* ba )
{
	size_t size;
	Bucket* bucket;

	// Allocate memory for the bucket
	size = sizeof(Bucket) + ba->itemSize * ba->bucketSize;
	bucket = (Bucket*)ba->alloc->memalloc( ba->alloc->userData, (unsigned int)size );
	if ( !bucket )
		return 0;

	// Add the bucket into the list of buckets.
	bucket->next = ba->buckets;
	ba->buckets = bucket;

	// Items are handed out lazily (bump allocation) from the new bucket, in address order.
	ba->cur = (unsigned char*)bucket + sizeof(Bucket);
	ba->end = ba->cur + ba->itemSize * ba->bucketSize;

	return 1;
}

struct BucketAlloc* createBucketAlloc( TESSalloc* alloc, const char* name,
									  unsigned int itemSize, unsigned int bucketSize )
{
	BucketAlloc* ba = (BucketAlloc*)alloc->memalloc( alloc->userData, sizeof(BucketAlloc) );

	ba->alloc = alloc;
	ba->name = name;
	ba->itemSize = itemSize;
	if ( ba->itemSize < sizeof(void*) )
		ba->itemSize = sizeof(void*);
	ba->bucketSize = bucketSize;
	ba->freelist = 0;
	ba->buckets = 0;
	ba->cur = 0;
	ba->end = 0;

	if ( !CreateBucket( ba ) )
	{
		alloc->memfree( alloc->userData, ba );
		return 0;
	}

	return ba;
}

void* bucketAlloc( struct BucketAlloc *ba )
{
	void *it;

	// Reuse freed items first.
	if ( ba->freelist )
	{
		it = ba->freelist;
		ba->freelist = *(void**)it;
		return it;
	}

	// Otherwise take the next untouched item, allocating a new bucket when the current one is used up.
	if ( ba->cur == ba->end )
	{
		if ( !CreateBucket( ba ) )
			return 0;
	}

	it = (void*)ba->cur;
	ba->cur += ba->itemSize;

	return it;
}

void bucketFree( struct BucketAlloc *ba, void *ptr )
{
#ifdef CHECK_BOUNDS
	int inBounds = 0;
	Bucket *bucket;

	// Check that the pointer is allocated with this allocator.
	bucket = ba->buckets;
	while ( bucket )
	{
		void *bucketMin = (void*)((unsigned char*)bucket + sizeof(Bucket));
		void *bucketMax = (void*)((unsigned char*)bucket + sizeof(Bucket) + ba->itemSize * ba->bucketSize);
		if ( ptr >= bucketMin && ptr < bucketMax )
		{
			inBounds = 1;
			break;
		}
		bucket = bucket->next;			
	}		

	if ( inBounds )
	{
		// Add the node in front of the free list.
		*(void**)ptr = ba->freelist;
		ba->freelist = ptr;
	}
	else
	{
		printf("ERROR! pointer 0x%p does not belong to allocator '%s'\n", ptr, ba->name);
	}
#else
	// Add the node in front of the free list.
	*(void**)ptr = ba->freelist;
	ba->freelist = ptr;
#endif
}

void deleteBucketAlloc( struct BucketAlloc *ba )
{
	TESSalloc* alloc = ba->alloc;
	Bucket *bucket = ba->buckets;
	Bucket *next;
	while ( bucket )
	{
		next = bucket->next;
		alloc->memfree( alloc->userData, bucket );
		bucket = next;
	}		
	ba->freelist = 0;
	ba->buckets = 0;
	ba->cur = 0;
	ba->end = 0;
	alloc->memfree( alloc->userData, ba );
}
