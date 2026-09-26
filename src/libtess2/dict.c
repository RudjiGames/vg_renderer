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
** Author: Eric Veach, July 1994.
*/

#include <stddef.h>
#include "tesselator.h"
#include "bucketalloc.h"
#include "sweep.h"
#include "dict.h"

/* really tessDictListNewDict */
Dict *dictNewDict( TESSalloc* alloc, void *frame, int (*leq)(TESStesselator *frame, ActiveRegion *key1, ActiveRegion *key2) )
{
	Dict *dict = (Dict *)alloc->memalloc( alloc->userData, sizeof( Dict ));
	DictNode *head;
	int i;

	if (dict == NULL) return NULL;

	head = &dict->head;

	head->key = NULL;
	head->next = head;
	head->prev = head;
	head->tower = &dict->headTower;
	dict->headTower.height = DICT_MAX_LEVELS;
	for( i = 0; i < DICT_MAX_LEVELS; ++i ) {
		dict->headTower.next[i] = head;
		dict->headTower.prev[i] = head;
	}
	dict->maxHeight = 1;
	dict->rng = 0x9E3779B9u; /* Fixed seed, the output must be deterministic */

	dict->frame = frame;
	dict->leq = leq;

	if (alloc->dictNodeBucketSize < 16)
		alloc->dictNodeBucketSize = 16;
	if (alloc->dictNodeBucketSize > 4096)
		alloc->dictNodeBucketSize = 4096;
	dict->nodePool = createBucketAlloc( alloc, "Dict", sizeof(DictNode), alloc->dictNodeBucketSize );
	dict->towerPool = createBucketAlloc( alloc, "DictTower", sizeof(DictTower), alloc->dictNodeBucketSize / 4 < 16 ? 16 : alloc->dictNodeBucketSize / 4 );
	if (dict->nodePool == NULL || dict->towerPool == NULL) {
		if (dict->nodePool) deleteBucketAlloc( dict->nodePool );
		if (dict->towerPool) deleteBucketAlloc( dict->towerPool );
		alloc->memfree( alloc->userData, dict );
		return NULL;
	}

	return dict;
}

/* really tessDictListDeleteDict */
void dictDeleteDict( TESSalloc* alloc, Dict *dict )
{
	deleteBucketAlloc( dict->towerPool );
	deleteBucketAlloc( dict->nodePool );
	alloc->memfree( alloc->userData, dict );
}

/* Previous node on the given level (the node must be on that level). */
static DictNode *dictPrevAt( DictNode *node, int level )
{
	return level == 0 ? node->prev : node->tower->prev[level];
}

/* Random height (1 + number of levels above 0): P(height > h) = 4^-(h-1). */
static int dictRandomHeight( Dict *dict )
{
	int height = 1;
	unsigned int r;

	/* xorshift32 */
	r = dict->rng;
	r ^= r << 13;
	r ^= r >> 17;
	r ^= r << 5;
	dict->rng = r;

	while( height < DICT_MAX_LEVELS && (r & 3) == 0 ) {
		++height;
		r >>= 2;
	}
	return height;
}

/* really tessDictListInsertBefore */
DictNode *dictInsertBefore( Dict *dict, DictNode *node, ActiveRegion *key )
{
	DictNode *newNode;
	DictTower *tower;
	int height, level;

	do {
		node = node->prev;
	} while( node->key != NULL && ! (*dict->leq)(dict->frame, node->key, key));

	newNode = (DictNode *)bucketAlloc( dict->nodePool );
	if (newNode == NULL) return NULL;

	newNode->key = key;
	newNode->next = node->next;
	node->next->prev = newNode;
	newNode->prev = node;
	node->next = newNode;
	newNode->tower = NULL;

	/* Promote the node to the upper levels of the skip list. If the tower can't be allocated, the node just stays
	* on level 0 (the dictionary is still valid, only searching is slower).
	*/
	height = dictRandomHeight( dict );
	if( height == 1 ) return newNode;

	tower = (DictTower *)bucketAlloc( dict->towerPool );
	if (tower == NULL) return newNode;

	tower->height = height;
	newNode->tower = tower;
	if( height > dict->maxHeight ) dict->maxHeight = height;

	/* The predecessor on level L is the closest node before the new node whose height is greater than L. It's found
	* by walking back on level L-1 from the predecessor on that level (expected O(1) steps per level; the head is on
	* all levels).
	*/
	for( level = 1; level < height; ++level ) {
		while( node->tower == NULL || node->tower->height <= level ) {
			node = dictPrevAt( node, level - 1 );
		}

		tower->prev[level] = node;
		tower->next[level] = node->tower->next[level];
		tower->next[level]->tower->prev[level] = newNode;
		node->tower->next[level] = newNode;
	}

	return newNode;
}

/* really tessDictListDelete */
void dictDelete( Dict *dict, DictNode *node ) /*ARGSUSED*/
{
	DictTower *tower = node->tower;
	int level;

	node->next->prev = node->prev;
	node->prev->next = node->next;

	if( tower != NULL ) {
		for( level = 1; level < tower->height; ++level ) {
			tower->prev[level]->tower->next[level] = tower->next[level];
			tower->next[level]->tower->prev[level] = tower->prev[level];
		}
		bucketFree( dict->towerPool, tower );
	}

	bucketFree( dict->nodePool, node );
}

/* really tessDictListSearch
* Returns the first node (in the list order) whose key is >= the given key (i.e. leq(key, node->key)), or the head
* if there's no such node. The upper levels of the skip list are used to skip the nodes whose key is < the given key.
*/
DictNode *dictSearch( Dict *dict, ActiveRegion *key )
{
	DictNode *node = &dict->head;
	DictNode *next;
	int level;

	for( level = dict->maxHeight - 1; level > 0; --level ) {
		for( ;; ) {
			next = node->tower->next[level];
			if( next->key == NULL || (*dict->leq)(dict->frame, key, next->key) ) break;
			node = next;
		}
	}

	do {
		node = node->next;
	} while( node->key != NULL && ! (*dict->leq)(dict->frame, key, node->key));

	return node;
}
